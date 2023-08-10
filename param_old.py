# -------------------------------------------------------------
# =================== Begin User Instructions =================
# -------------------------------------------------------------
source_name = "MACSJ0329.6-0211"       # The cluster name
obsids = "3257"                        # The obsids to be considered "id1,id2,id3,..."
z = 0.45                               # The cluster redshift

R500 = 950.0         # A first guess of the cluster R500 radius
use_peak = True      # Will use the large-scale centroid as a deprojection center if False
fixed_coord = None   # None or [R.A.,Dec.] in degree | set use_peak to True if [R.A.,Dec.]
fast_annuli = True   # Set to True if you want to use the McDonald+17 binning definition for the SB profile

Ysz = [6e-05, 1e-05]


single_ann_spec = False  # Set to True to consider only one annulus between 0.15R500 and R500 for the spectrum extraction


file_ACCEPT = None  # Set the path to the ascii file downloaded from the ACCEPT database for comparison


compute_Lcool = True      # Set to True if you want to compute the cooling luminosity
tcool_th = 7.7            # The cooling time threshold to be considered to compute the cooling radius
input_XSZ_file = None     # The path to the fits file containing the ICM profiles estimated from a joint X-ray/SZ analysis
do_err = True             # Set to True to compute the error bar on the cooling luminosity

# -------------------------------------------------------------
# =================== End User Instructions ===================
# -------------------------------------------------------------


tab_obsid = obsids.split(",")
if len(tab_obsid) > 1:
    multiobs = True
else:
    multiobs = False
