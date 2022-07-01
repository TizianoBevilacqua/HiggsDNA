import awkward
import vector

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import lepton_selections, jet_selections, gen_jet_selections
from higgs_dna.utils import awkward_utils, misc_utils

DUMMY_VALUE = -999.

DEFAULT_OPTIONS = {
    "higgs" : {
        "pt" : 200.0
    },
    "jets" : {
        "n_jets" : [0, 1],
        "m" : 0.2
    },
    "jets" : {
        "pt" : 30.0,
        "eta" : 2.5,
        "dr_photons" : 0.4,
        "dr_electrons" : 0.4,
        "dr_muons" : 0.4,
    },
    "gen_jets" : {
        "pt" : 0.0,
        "eta" : 2.5,
        "dr_photons" : 0.4,
        "dr_electrons" : 0.001,
        "dr_muons" : 0.001,
    },
    "photon_id" : -0.9,
    "btag_wp" : {
        "2016" : 0.3093,
        "2017" : 0.3040,
        "2018" : 0.2783
    }
}   

class ggH_STXS_split_tagger(Tagger):
    """
    ggH STXS splitter tagger built following AN2019_259 division
    """
    def __init__(self, name, options = {}, is_data = None, year = None):
        super(ggH_STXS_split_tagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )

    def calculate_selection(self, events):

        BSM = (events.Diphoton.pt >= 200)
        Zerojets_lowPt           = (events.n_jets == 0) & (events.Diphoton.pt < 10)   
        Zerojets_highPt          = (events.n_jets == 0) & (events.Diphoton.pt >= 10)  & (events.Diphoton.pt < 200)
        Onejets_lowPt            = (events.n_jets == 1) & (events.Diphoton.pt < 60)   
        Onejets_medPt            = (events.n_jets == 1) & (events.Diphoton.pt >= 60)  & (events.Diphoton.pt < 120)
        Onejets_highPt           = (events.n_jets == 1) & (events.Diphoton.pt >= 120) & (events.Diphoton.pt < 200)

        jet_1_split = awkward.Array({"pt": events.jet_1_pt, "eta": events.jet_1_eta, "phi": events.jet_1_phi, "mass": events.jet_1_mass})
        jet_1 = awkward.Array(jet_1_split, with_name = "Momentum4D")
        jet_2_split = awkward.Array({"pt": events.jet_2_pt, "eta": events.jet_2_eta, "phi": events.jet_2_phi, "mass": events.jet_2_mass})
        jet_2 = awkward.Array(jet_2_split, with_name = "Momentum4D")
        dijet = jet_1 + jet_2

        GETwojets_lowmjj_lowPt  = (events.n_jets >= 2) & (dijet.mass < 350) & (events.Diphoton.pt < 60)
        GETwojets_lowmjj_medPt  = (events.n_jets >= 2) & (dijet.mass < 350) & (events.Diphoton.pt >= 60)  & (events.Diphoton.pt < 120)
        GETwojets_lowmjj_highPt = (events.n_jets >= 2) & (dijet.mass < 350) & (events.Diphoton.pt >= 120) & (events.Diphoton.pt < 200)

        VBF_Twojets_lowmjj      = (events.n_jets == 2) & (dijet.mass >= 350) & (dijet.mass < 700) & (events.Diphoton.pt < 200)
        VBF_Twojets_highmjj     = (events.n_jets == 2) & (dijet.mass >= 700)                      & (events.Diphoton.pt < 200)
        VBF_Threejets_lowmjj    = (events.n_jets >= 3) & (dijet.mass >= 350) & (dijet.mass < 700) & (events.Diphoton.pt < 200)
        VBF_Threejets_highmjj   = (events.n_jets >= 3) & (dijet.mass >= 700)                      & (events.Diphoton.pt < 200)

        events["is_ggH_BSM"]                     = BSM
        events["is_ggH_Zerojets_lowPt"]          = Zerojets_lowPt
        events["is_ggH_Zerojets_highPt"]         = Zerojets_highPt
        events["is_ggH_Onejets_lowPt"]           = Onejets_lowPt
        events["is_ggH_Onejets_medPt"]           = Onejets_medPt
        events["is_ggH_Onejets_highPt"]          = Onejets_highPt
        events["is_ggH_GETwojets_lowmjj_lowPt"]  = GETwojets_lowmjj_lowPt
        events["is_ggH_GETwojets_lowmjj_medPt"]  = GETwojets_lowmjj_medPt
        events["is_ggH_GETwojets_lowmjj_highPt"] = GETwojets_lowmjj_highPt
        events["is_ggH_VBF_Twojets_lowmjj"]      = VBF_Twojets_lowmjj
        events["is_ggH_VBF_Twojets_highmjj"]     = VBF_Twojets_highmjj
        events["is_ggH_VBF_Threejets_lowmjj"]    = VBF_Threejets_lowmjj
        events["is_ggH_VBF_Threejets_highmjj"]   = VBF_Threejets_highmjj

        presel_cut = (Zerojets_lowPt | Zerojets_highPt 
                    | Onejets_lowPt  | Onejets_medPt | Onejets_highPt 
                    | GETwojets_lowmjj_lowPt | GETwojets_lowmjj_medPt | GETwojets_lowmjj_highPt 
                    | VBF_Twojets_lowmjj | VBF_Twojets_highmjj | VBF_Threejets_lowmjj | VBF_Threejets_highmjj 
                    | BSM)
                    
        self.register_cuts(
            names = ["ggh_all"],
            results = [presel_cut]
        )

        return presel_cut, events