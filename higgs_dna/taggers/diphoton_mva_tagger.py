import awkward
import vector
import xgboost
import numpy
import numba

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.utils import awkward_utils, misc_utils

DEFAULT_OPTIONS = {
    "bdt_file" : "data/altDiphoModel_coffea.json", # if running on condor, this file needs to be placed somewhere under higgs_dna/ so that it is included in the tar file. We probably want to think of a better long term solution for this.
    "bdt_features" : [
        "sigmaMrv", "sigmaMwv", "vtxProb",
        ("LeadPhoton", "pt_mgg"), ("LeadPhoton", "eta"), ("LeadPhoton", "mvaID"), ("SubleadPhoton", "pt_mgg"), ("SubleadPhoton", "eta"), ("SubleadPhoton", "mvaID"),
        ("Diphoton", "cos_dPhi")
    ],
    "bdt_cuts" : [0.9898, 0.882222, 0.0]
}

DUMMY_VALUE = -999.

class diphoton_mva_tagger(Tagger):
    """
    diphoton_mva_tagger for th cH->HToGG analysis.
    """
    def __init__(self, name, options = {}, is_data = None, year = None):
        super(diphoton_mva_tagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )


    def calculate_selection(self, events): 

        def calc_displacement(photons: awkward.Array, events: awkward.Array) -> awkward.Array:
            x = photons.x_calo - events.PV_x
            y = photons.y_calo - events.PV_y
            z = photons.z_calo - events.PV_z
            return awkward.zip({"x": x, "y": y, "z": z}, with_name="Vector3D")

        #calculating mass resolution in the correct vertex hipotesis
        sigmaEoE_Lead = events.LeadPhoton.energyErr /  events.LeadPhoton.energy 
        sigmaEoE_Sublead = events.SubleadPhoton.energyErr /  events.SubleadPhoton.energy
        sigma_m_rv = 0.5 * numpy.sqrt(sigmaEoE_Lead ** 2 + sigmaEoE_Sublead ** 2)

        #calculating mass resolution in the vrong vertex hipotesis
        #the calculation for sigma_vtx  is synched from flashgg
        v_lead = calc_displacement(events.LeadPhoton, events)
        v_sublead = calc_displacement(events.SubleadPhoton, events)

        r1 = numpy.sqrt(v_lead.x ** 2 + v_lead.y ** 2 + v_lead.z ** 2)
        r2 = numpy.sqrt(v_sublead.x ** 2 + v_sublead.y ** 2 + v_sublead.z ** 2)

        p_lead = v_lead.unit() * events.LeadPhoton.energyRaw #normalizaed to 1 * rawenergy
        p_lead["energy"] = events.LeadPhoton.energyRaw
        p_lead = awkward.with_name(p_lead, "Momentum4D") #four-vector wrapping
        p_sublead = v_sublead.unit() * events.SubleadPhoton.energyRaw
        p_sublead["energy"] = events.SubleadPhoton.energyRaw
        p_sublead = awkward.with_name(p_sublead, "Momentum4D")

        sech_lead = 1.0 / numpy.cosh(p_lead.eta) #angular quantities
        sech_sublead = 1.0 / numpy.cosh(p_sublead.eta)
        tanh_lead = numpy.tanh(p_lead.eta)
        tanh_sublead = numpy.cos(p_sublead.theta)
        tanh_sublead = numpy.tanh(p_sublead.eta)

        cos_dphi = numpy.cos(p_lead.deltaphi(p_sublead)) 

        numerator_lead = sech_lead * (sech_lead * tanh_sublead - tanh_lead * sech_sublead * cos_dphi)
        numerator_sublead = sech_sublead * (sech_sublead * tanh_lead - tanh_sublead * sech_lead * cos_dphi)
    
        denominator = 1.0 - tanh_lead * tanh_sublead - sech_lead * sech_sublead * cos_dphi
    
        sigma_vtx = (0.5 * (-numpy.sqrt(2.0) * events.BeamSpot_sigmaZ / denominator) * (numerator_lead / r1 + numerator_sublead / r2))
        sigma_m_wv = numpy.sqrt(sigma_m_rv ** 2 + sigma_vtx ** 2)

        #additional variables to compensate for the absence of the vtxProb MVA score

        #vtx_prob = awkward.full_like(sigma_m_rv, 0.999)  # !!!! placeholder !!!!
        vtx_prob = 2*sigma_m_rv/(sigma_m_rv+sigma_m_wv)  # !!!! placeholder !!!!

        #z coordinate of primary vertices other than the main one
        #padded to have at least 3 entry for each event (useful for slicing)
        OtherPV_z = awkward.to_numpy(awkward.fill_none(awkward.pad_none(events.OtherPV.z, 3, axis=1), DUMMY_VALUE))
        PV_z = awkward.to_numpy(events.PV_z)
        events.OtherPV.z = awkward.from_numpy(OtherPV_z)
        #reshaping to match OtherPV_z
        PV_z = numpy.full_like(numpy.arange(3*len(PV_z)).reshape(len(PV_z), 3), 1, dtype=float)
        PV_z[:,0] = PV_z[:,0]*events.PV_z
        PV_z[:,1] = PV_z[:,1]*events.PV_z
        PV_z[:,2] = PV_z[:,2]*events.PV_z
        #z distance of the first three PVs from the main one 
        events["OtherPV_dZ_0"] = awkward.from_numpy(numpy.abs(PV_z - OtherPV_z))
        
        dZ = awkward_utils.add_field(
                events = events,
                name = "dZ",
                data = events.OtherPV_dZ_0
        ) 

        for i in range(len(events.OtherPV.z[0])):
                awkward_utils.add_field(
                    events = events,
                    name = "%s_%d" % ("dZ", i+1),
                    data = awkward.fill_none(events.OtherPV_dZ_0[:,i], DUMMY_VALUE),
                    overwrite = True
                )

        events["vtxProb"] = vtx_prob
        events["sigmaMrv"] = sigma_m_rv
        events["sigmaMwv"] = sigma_m_wv
  
        events["PV_score"] = events.PV_score
        events["PV_chi2"] = events.PV_chi2
        events["nPV"] = events.PV_npvs

        events["event"] = events.event
        events["run"] = events.run
       
        #counts = awkward.num(events.Diphotons, axis=-1)
        #bdt_inputs = numpy.column_stack( [awkward.to_numpy(awkward.flatten(events[name])) for name in var_order] )

        # Initialize BDT 

        def _get_gzip(fname: str) -> bytearray:
            return bytearray(gzip.open(fname, "rb").read())


        def _get_lzma(fname: str) -> bytearray:
            return bytearray(lzma.open(fname, "rb").read())

        _magics = {
            b"\x1f\x8b": _get_gzip,
            b"\xfd7": _get_lzma,
        }

        bdt = xgboost.Booster()
        fname = misc_utils.expand_path(self.options["bdt_file"])
        with open(fname, "rb") as f:
            magic = f.read(2)
            opener = _magics.get(magic, lambda x: x)
        bdt.load_model(opener(fname))
       
        # Convert events to proper format for xgb
        events_bdt = awkward.values_astype(events, numpy.float64)

        bdt_features = []
        for x in self.options["bdt_features"]:
            if isinstance(x, tuple):
                name_flat = "_".join(x)
                events_bdt[name_flat] = events_bdt[x]
                bdt_features.append(name_flat)
            else:
                bdt_features.append(x)

        features_bdt = awkward.to_numpy( events_bdt[bdt_features] )

        features_bdt = xgboost.DMatrix(
                features_bdt.view((float, len(features_bdt.dtype.names)))
        )

        # Calculate BDT score
        events["bdt_score"] = bdt.predict(features_bdt)

        # Calculate OR of all BDT cuts
        presel_cut = events.run >= 0 # dummy all True
        #for cut in sr_cuts:
        #    presel_cut = presel_cut | cut

        return presel_cut, events
