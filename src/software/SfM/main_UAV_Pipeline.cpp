
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include <thread>
#include <chrono>

#include <cereal/archives/json.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h> 
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#ifdef __cplusplus 
}
#endif

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include "openMVG/cameras/cameras.hpp"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "openMVG/exif/exif_IO_EasyExif.hpp"
#include "openMVG/exif/sensor_width_database/ParseDatabase.hpp"
#include "openMVG/features/akaze/image_describer_akaze_io.hpp"
#include "openMVG/features/descriptor.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/geodesy/geodesy.hpp"
#include "openMVG/graph/graph.hpp"
#include "openMVG/graph/graph_stats.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/GeometricFilter.hpp"
#include "openMVG/matching_image_collection/F_ACRobust.hpp"
#include "openMVG/matching_image_collection/E_ACRobust.hpp"
#include "openMVG/matching_image_collection/E_ACRobust_Angular.hpp"
#include "openMVG/matching_image_collection/Eo_Robust.hpp"
#include "openMVG/matching_image_collection/H_ACRobust.hpp"
#include "openMVG/matching_image_collection/Pair_Builder.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching/pairwiseAdjacencyDisplay.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_utils.hpp"
#include "openMVG/sfm/sfm_view.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/sfm/sfm_report.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/sfm_global_engine_relative_motions.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp"
#include "openMVG/sfm/pipelines/sequential/sequential_SfM.hpp"
#include "openMVG/sfm/pipelines/sequential/sequential_SfM2.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerStellar.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializer.hpp"
#include "openMVG/sfm/sfm_data_colorization.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/types.hpp"
#include "nonFree/sift/SIFT_describer_io.hpp"
#include "software/SfM/SfMPlyHelper.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "sfm_video_view.hpp"
#include "sfm_video_view_io.hpp"
#include "thread_safe_queue.hpp"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::exif;
using namespace openMVG::geodesy;
using namespace openMVG::image;
using namespace openMVG::sfm;
using namespace openMVG::features;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace openMVG::matching_image_collection;
using namespace std;

// Shared queue for video batches
static thread_safe_queue<SfM_Data*> vid_queue;
static atomic<bool> vid_done(false);
static int BATCH_SIZE = 50;

enum EGeometricModel
{
  FUNDAMENTAL_MATRIX = 0,
  ESSENTIAL_MATRIX   = 1,
  HOMOGRAPHY_MATRIX  = 2,
  ESSENTIAL_MATRIX_ANGULAR = 3,
  ESSENTIAL_MATRIX_ORTHO = 4,
  ESSENTIAL_MATRIX_UPRIGHT = 5
};

enum EPairMode
{
  PAIR_EXHAUSTIVE = 0,
  PAIR_CONTIGUOUS = 1,
  PAIR_FROM_FILE  = 2
};

enum Pipeline
{
  PIPELINE_HYBRID = 0,
  PIPELINE_GLOBAL = 1,
  PIPELINE_SEQUENTIAL = 2
};

struct Matches_Provider_Video : public openMVG::sfm::Matches_Provider {
public:

  explicit Matches_Provider_Video
  (const SfM_Data & sfm_data, PairWiseMatches & matches) : Matches_Provider()
  {
    pairWise_matches_.swap(matches);
  }
};

struct Features_Provider_Video : public openMVG::sfm::Features_Provider {
public:

  explicit Features_Provider_Video
  (const SfM_Data & sfm_data, const std::shared_ptr<Regions_Provider> & regions)
  {
    for (Views::const_iterator iter = sfm_data.GetViews().begin();
      iter != sfm_data.GetViews().end();++iter)
    {
      std::shared_ptr<features::Regions> region = regions->get(iter->second->id_view);
      feats_per_view[iter->second->id_view] = region->GetRegionsPositions();
    }
  } 
};

struct Regions_Provider_Video : public sfm::Regions_Provider {
public:

  explicit Regions_Provider_Video
  (
    const Views & views,
    std::unique_ptr<features::Regions>& region_type
  ): Regions_Provider()
  {
    region_type_.reset(region_type->EmptyClone());
    for (Views::const_iterator iter = views.begin();
      iter != views.end(); ++iter)
    {
      VideoView* view = (VideoView*)iter->second.get();
      cache_[view->id_view] = std::move(view->regions);
    }
  }
};

struct Settings {
public:

  void make_options(CmdLine &cmd)
  {
    cmd.add( make_option('i', image_dir, "imageDirectory") );
    cmd.add( make_option('d', file_database, "sensorWidthDatabase") );
    cmd.add( make_option('o', output_dir, "outputDirectory") );
    cmd.add( make_option('f', focal_pixels, "focal") );
    cmd.add( make_option('k', k_matrix, "intrinsics") );
    cmd.add( make_option('c', i_user_camera_model, "camera_model") );
    cmd.add( make_option('g', group_camera_model, "group_camera_model") );
    cmd.add( make_switch('P', "use_motion_priors") );
    cmd.add( make_option('m', gps_xyz_method, "gps_to_xyz_method") );
    cmd.add( make_option('v', vid_device, "videoDevice") );

#ifdef OPENMVG_USE_OPENMP
    cmd.add( make_option('n', num_threads, "numThreads") );
#endif

    cmd.add( make_option('M', image_describer_method, "describerMethod") );
    cmd.add( make_option('u', up_right, "upright") );
    cmd.add( make_option('p', feature_preset, "describerPreset") );

    cmd.add( make_option('r', dist_ratio, "ratio") );
    cmd.add( make_option('e', geometric_model, "geometric_model") );
    cmd.add( make_option('E', matching_video_mode, "video_mode_matching") );
    cmd.add( make_option('l', predefined_pair_list, "pair_list") );
    cmd.add( make_option('n', nearest_matching_method, "nearest_matching_method") );
    cmd.add( make_option('G', guided_matching, "guided_matching") );
    cmd.add( make_option('I', imax_iteration, "max_iteration") );
    cmd.add( make_option('C', ui_max_cache_size, "cache_size") );

    cmd.add( make_option('A', s_intrinsic_refinement_options, "refineIntrinsics") );
    cmd.add( make_option('T', triangulation_method, "triangulation_method"));
    cmd.add( make_option('R', resection_method, "resection_method"));
    cmd.add( make_option('X', rotation_averaging_method, "rotationAveraging") );
    cmd.add( make_option('Z', translation_averaging_method, "translationAveraging") );
    cmd.add( make_option('x', pipeline, "pipeline") );
  }

  void output_usage(const char* args)
  {
    std::cerr << "Usage: " << args << '\n'
    << "[-i|--imageDirectory]\n"
    << "[-d|--sensorWidthDatabase]\n"
    << "[-o|--outputDirectory]\n"
    << "[-f|--focal] (pixels)\n"
    << "[-k|--intrinsics] Kmatrix: \"f;0;ppx;0;f;ppy;0;0;1\"\n"
    << "[-c|--camera_model] Camera model type:\n"
    << "\t" << static_cast<int>(PINHOLE_CAMERA) << ": Pinhole\n"
    << "\t" << static_cast<int>(PINHOLE_CAMERA_RADIAL1) << ": Pinhole radial 1\n"
    << "\t" << static_cast<int>(PINHOLE_CAMERA_RADIAL3) << ": Pinhole radial 3 (default)\n"
    << "\t" << static_cast<int>(PINHOLE_CAMERA_BROWN) << ": Pinhole brown 2\n"
    << "\t" << static_cast<int>(PINHOLE_CAMERA_FISHEYE) << ": Pinhole with a simple Fish-eye distortion\n"
    << "\t" << static_cast<int>(CAMERA_SPHERICAL) << ": Spherical camera\n"
    << "[-g|--group_camera_model]\n"
    << "\t 0-> each view have it's own camera intrinsic parameters,\n"
    << "\t 1-> (default) view can share some camera intrinsic parameters\n"
    << "\n"
    << "[-P|--use_pose_prior] Use pose prior if GPS EXIF pose is available"
    << "[-m|--gps_to_xyz_method] XZY Coordinate system:\n"
    << "[-v|--videoDevice] Video device for the camera:\n"
    << "\t 0: ECEF (default)\n"
    << "\t 1: UTM\n"
    << "[-M|--describerMethod]\n"
    << "  (method to use to describe an image):\n"
    << "   SIFT (default),\n"
    << "   SIFT_ANATOMY,\n"
    << "   AKAZE_FLOAT: AKAZE with floating point descriptors,\n"
    << "   AKAZE_MLDB:  AKAZE with binary descriptors\n"
    << "[-u|--upright] Use Upright feature 0 or 1\n"
    << "[-p|--describerPreset]\n"
    << "  (used to control the Image_describer configuration):\n"
    << "   NORMAL (default),\n"
    << "   HIGH,\n"
    << "   ULTRA: !!Can take long time!!\n"
#ifdef OPENMVG_USE_OPENMP
    << "[-n|--numThreads] number of parallel computations\n"
#endif
    << "[-r|--ratio] Distance ratio to discard non meaningful matches\n"
    << "   0.8: (default).\n"
    << "[-e|--geometric_model]\n"
    << "  (pairwise correspondences filtering thanks to robust model estimation):\n"
    << "   f: (default) fundamental matrix,\n"
    << "   e: essential matrix,\n"
    << "   h: homography matrix.\n"
    << "   a: essential matrix with an angular parametrization,\n"
    << "   o: orthographic essential matrix.\n"
    << "   u: upright essential matrix.\n"
    << "[-E|--video_mode_matching]\n"
    << "  (sequence matching with an overlap of X images)\n"
    << "   X: with match 0 with (1->X), ...]\n"
    << "   2: will match 0 with (1,2), 1 with (2,3), ...\n"
    << "   3: will match 0 with (1,2,3), 1 with (2,3,4), ...\n"
    << "[-l]--pair_list] file\n"
    << "[-n|--nearest_matching_method]\n"
    << "  AUTO: auto choice from regions type,\n"
    << "  For Scalar based regions descriptor:\n"
    << "    BRUTEFORCEL2: L2 BruteForce matching,\n"
    << "    HNSWL2: L2 Approximate Matching with Hierarchical Navigable Small World graphs,\n"
    << "    ANNL2: L2 Approximate Nearest Neighbor matching,\n"
    << "    CASCADEHASHINGL2: L2 Cascade Hashing matching.\n"
    << "    FASTCASCADEHASHINGL2: (default)\n"
    << "      L2 Cascade Hashing with precomputed hashed regions\n"
    << "     (faster than CASCADEHASHINGL2 but use more memory).\n"
    << "  For Binary based descriptor:\n"
    << "    BRUTEFORCEHAMMING: BruteForce Hamming matching.\n"
    << "[-G|--guided_matching]\n"
    << "  use the found model to improve the pairwise correspondences.\n"
    << "[-C|--cache_size]\n"
    << "  Use a regions cache (only cache_size regions will be stored in memory)\n"
    << "  If not used, all regions will be load in memory."
    << "[-A|--refineIntrinsics] Intrinsic parameters refinement option\n"
    << "\t ADJUST_ALL -> refine all existing parameters (default) \n"
    << "\t NONE -> intrinsic parameters are held as constant\n"
    << "\t ADJUST_FOCAL_LENGTH -> refine only the focal length\n"
    << "\t ADJUST_PRINCIPAL_POINT -> refine only the principal point position\n"
    << "\t ADJUST_DISTORTION -> refine only the distortion coefficient(s) (if any)\n"
    << "\t -> NOTE: options can be combined thanks to '|'\n"
    << "\t ADJUST_FOCAL_LENGTH|ADJUST_PRINCIPAL_POINT\n"
    <<      "\t\t-> refine the focal length & the principal point position\n"
    << "\t ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\n"
    <<      "\t\t-> refine the focal length & the distortion coefficient(s) (if any)\n"
    << "\t ADJUST_PRINCIPAL_POINT|ADJUST_DISTORTION\n"
    <<      "\t\t-> refine the principal point position & the distortion coefficient(s) (if any)\n"
    << "[-T|--triangulation_method] triangulation method (default=" << triangulation_method << "):\n"
    << "\t" << static_cast<int>(ETriangulationMethod::DIRECT_LINEAR_TRANSFORM) << ": DIRECT_LINEAR_TRANSFORM\n"
    << "\t" << static_cast<int>(ETriangulationMethod::L1_ANGULAR) << ": L1_ANGULAR\n"
    << "\t" << static_cast<int>(ETriangulationMethod::LINFINITY_ANGULAR) << ": LINFINITY_ANGULAR\n"
    << "\t" << static_cast<int>(ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT) << ": INVERSE_DEPTH_WEIGHTED_MIDPOINT\n"
    << "[-R|--resection_method] resection/pose estimation method (default=" << resection_method << "):\n"
    << "\t" << static_cast<int>(resection::SolverType::DLT_6POINTS) << ": DIRECT_LINEAR_TRANSFORM 6Points | does not use intrinsic data\n"
    << "\t" << static_cast<int>(resection::SolverType::P3P_KE_CVPR17) << ": P3P_KE_CVPR17\n"
    << "\t" << static_cast<int>(resection::SolverType::P3P_KNEIP_CVPR11) << ": P3P_KNEIP_CVPR11\n"
    << "\t" << static_cast<int>(resection::SolverType::P3P_NORDBERG_ECCV18) << ": P3P_NORDBERG_ECCV18\n"
    << "\t" << static_cast<int>(resection::SolverType::UP2P_KUKELOVA_ACCV10)  << ": UP2P_KUKELOVA_ACCV10 | 2Points | upright camera\n"
    << "[-X|--rotationAveraging]\n"
    << "\t 1 -> L1 minimization\n"
    << "\t 2 -> L2 minimization (default)\n"
    << "[-Z|--translationAveraging]:\n"
    << "\t 1 -> L1 minimization\n"
    << "\t 2 -> L2 minimization of sum of squared Chordal distances\n"
    << "\t 3 -> SoftL1 minimization (default)\n"
    << "[-x|--pipeline] Pipline to use (default=" << pipeline << ")\n"
    << std::endl;
  }

  void display(const char* args)
  {
    std::cout << " You called : " <<std::endl
              << args << std::endl
              << "Image Directory - " << image_dir << std::endl
              << "Sensor Width Database - " << file_database << std::endl
              << "Output Directory - " << output_dir << std::endl
              << "Focal Length - " << focal << std::endl
              << "PPX - " << ppx << ", PPY - " << ppy << endl
 	      << "K Matrix - " << k_matrix << std::endl
              << "I Camera Model - " << i_user_camera_model << std::endl
              << "Group Camera Model - " << group_camera_model << std::endl
              << "Describer Method - " << image_describer_method << std::endl
              << "Up Right - " << up_right << std::endl
              << "Describer Preset - " << (feature_preset.empty() ? "NORMAL" : feature_preset) << std::endl
#ifdef OPENMVG_USE_OPENMP
              << "Number Threads - " << num_threads << std::endl
#endif
              << "Video Device - " << vid_device << std::endl
              << "Ratio - " << dist_ratio << std::endl
              << "Geometric Model - " << geometric_model << std::endl
              << "Geomtric Model To Compute - " << e_geometric_model_to_compute << endl
	      << "Video Mode Matching - " << matching_video_mode << std::endl
              << "Pair List - " << predefined_pair_list << std::endl
              << "Nearest Matching Method - " << nearest_matching_method << std::endl
              << "Guided Matching - " << guided_matching << std::endl
              << "Triangulation Method - " << triangulation_method << std::endl
              << "Pipeline - " << pipeline << endl
	      << "GPS XYZ Method - " << gps_xyz_method << endl
	      << "Max Iteration - " << imax_iteration << endl
	      << "Intrinsic Refinment Options - " << s_intrinsic_refinement_options << endl
	      << "Use Motion Priors - " << use_motion_priors << endl
	      << "Resection Method - " << resection_method << endl
	      << "Rotation Averaging Method - " << rotation_averaging_method << endl
	      << "Tranlsation Averaging Method - " << translation_averaging_method << endl
	      << "E Camera Model - " << e_user_camera_model << endl
              << "E Pair Mode - " << e_pair_mode << endl
	      << "Cache Size - " << ((ui_max_cache_size == 0) ? "unlimited" : std::to_string(ui_max_cache_size)) << std::endl;
  }
 
  /// Check that Kmatrix is a string like "f;0;ppx;0;f;ppy;0;0;1"
  /// With f,ppx,ppy as valid numerical value
  bool checkIntrinsicStringValidity(const std::string & Kmatrix, double & focal, double & ppx, double & ppy)
  {
    std::vector<std::string> vec_str;
    stl::split(Kmatrix, ';', vec_str);
    if (vec_str.size() != 9)  {
      std::cerr << "\n Missing ';' character" << std::endl;
      return false;
    }
    // Check that all K matrix value are valid numbers
    for (size_t i = 0; i < vec_str.size(); ++i) {
      double readvalue = 0.0;
      std::stringstream ss;
      ss.str(vec_str[i]);
      if (! (ss >> readvalue) )  {
        std::cerr << "\n Used an invalid not a number character" << std::endl;
        return false;
      }
      if (i==0) focal = readvalue;
      if (i==2) ppx = readvalue;
      if (i==5) ppy = readvalue;
    }
    return true;
  }

  /// Check string of prior weights
  std::pair<bool, Vec3> checkPriorWeightsString
  (
    const std::string &sWeights
  )
  {
    std::pair<bool, Vec3> val(true, Vec3::Zero());
    std::vector<std::string> vec_str;
    stl::split(sWeights, ';', vec_str);
    if (vec_str.size() != 3)
    {
      std::cerr << "\n Missing ';' character in prior weights" << std::endl;
      val.first = false;
    }
    // Check that all weight values are valid numbers
    for (size_t i = 0; i < vec_str.size(); ++i)
    {
      double readvalue = 0.0;
      std::stringstream ss;
      ss.str(vec_str[i]);
      if (! (ss >> readvalue) )  {
        std::cerr << "\n Used an invalid not a number character in local frame origin" << std::endl;
        val.first = false;
      }
      val.second[i] = readvalue;
    }
    return val;
  }

  features::EDESCRIBER_PRESET stringToEnum(const std::string & sPreset)
  { 
    features::EDESCRIBER_PRESET preset;
    if (sPreset == "NORMAL")
      preset = features::NORMAL_PRESET;
    else
    if (sPreset == "HIGH")
      preset = features::HIGH_PRESET;
    else
    if (sPreset == "ULTRA")
      preset = features::ULTRA_PRESET;
    else
      preset = features::EDESCRIBER_PRESET(-1);
    return preset;
  }

  bool init(CmdLine &cmd)
  {
    e_user_camera_model = EINTRINSIC(i_user_camera_model);

    if ( !stlplus::folder_exists( image_dir ) )
    {
      std::cerr << "\nThe input directory doesn't exist" << std::endl;
      return false;
    }

    if (output_dir.empty())
    {
      std::cerr << "\nInvalid output directory" << std::endl;
      return false;
    }

    if ( !stlplus::folder_exists( output_dir ) )
    {
      if ( !stlplus::folder_create( output_dir ))
      {
        std::cerr << "\nCannot create output directory" << std::endl;
        return false;
      }
    }

    if (k_matrix.size() > 0 &&
      !checkIntrinsicStringValidity(k_matrix, focal, ppx, ppy) )
    {
      std::cerr << "\nInvalid K matrix input" << std::endl;
      return false;
    }

    if (k_matrix.size() > 0 && focal_pixels != -1.0)
    {
      std::cerr << "\nCannot combine -f and -k options" << std::endl;
      return false;
    }
   
    std::vector<Datasheet> vec_database;
    if (!file_database.empty())
    {
      if ( !parseDatabase( file_database, vec_database ) )
      {
        std::cerr
         << "\nInvalid input database: " << file_database
         << ", please specify a valid file." << std::endl;
        return false;
      }
    }

    // Create the desired Image_describer method.
    // Don't use a factory, perform direct allocation
    if (image_describer_method == "SIFT")
    {
      image_describer.reset(new SIFT_Image_describer
        (SIFT_Image_describer::Params(), !up_right));
    }
    else
    if (image_describer_method == "SIFT_ANATOMY")
    {
      image_describer.reset(
        new SIFT_Anatomy_Image_describer(SIFT_Anatomy_Image_describer::Params()));
    }
    else
    if (image_describer_method == "AKAZE_FLOAT")
    {
      image_describer = AKAZE_Image_describer::create
        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MSURF), !up_right);
    }
    else
    if (image_describer_method == "AKAZE_MLDB")
    {
      image_describer = AKAZE_Image_describer::create
        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MLDB), !up_right);
    }

    if (!image_describer)
    {
      std::cerr << "Cannot create the designed Image_describer:"
        << image_describer_method << "." << std::endl;
      return false;
    }
    else
    {
      if (!feature_preset.empty())
      if (!image_describer->Set_configuration_preset(stringToEnum(feature_preset)))
      {
        std::cerr << "Preset configuration failed." << std::endl;
        return false;
      }
    }

    e_pair_mode = (matching_video_mode == -1 ) ? PAIR_EXHAUSTIVE : PAIR_CONTIGUOUS;

    if (predefined_pair_list.length()) {
      e_pair_mode = PAIR_FROM_FILE;
      if (matching_video_mode > 0) {
        std::cerr << "\nIncompatible options: --videoModeMatching and --pairList" << std::endl;
        return false;
      }
    }

    e_geometric_model_to_compute = FUNDAMENTAL_MATRIX;
    switch (geometric_model[0])
    {
      case 'f': case 'F':
        e_geometric_model_to_compute = FUNDAMENTAL_MATRIX;
      break;
      case 'e': case 'E':
        e_geometric_model_to_compute = ESSENTIAL_MATRIX;
      break;
      case 'h': case 'H':
        e_geometric_model_to_compute = HOMOGRAPHY_MATRIX;
      break;
      case 'a': case 'A':
        e_geometric_model_to_compute = ESSENTIAL_MATRIX_ANGULAR;
      break;
      case 'o': case 'O':
        e_geometric_model_to_compute = ESSENTIAL_MATRIX_ORTHO;
      break;
      case 'u': case 'U':
        e_geometric_model_to_compute = ESSENTIAL_MATRIX_UPRIGHT;
      break;
      default:
        std::cerr << "Unknown geometric model" << std::endl;
        return false;
    }

    if ( !isValid(static_cast<ETriangulationMethod>(triangulation_method))) {
      std::cerr << "\n Invalid triangulation method" << std::endl;
      return false;
    }

    intrinsic_refinement_options = cameras::StringTo_Intrinsic_Parameter_Type(s_intrinsic_refinement_options);
    if (intrinsic_refinement_options == static_cast<cameras::Intrinsic_Parameter_Type>(0) )
    {
      std::cerr << "Invalid input for Bundle Adjusment Intrinsic parameter refinement option" << std::endl;
      return false;
    }

    if (rotation_averaging_method < ROTATION_AVERAGING_L1 ||
        rotation_averaging_method > ROTATION_AVERAGING_L2 )  {
      std::cerr << "\n Rotation averaging method is invalid" << std::endl;
      return false;
    }

    if (translation_averaging_method < TRANSLATION_AVERAGING_L1 ||
        translation_averaging_method > TRANSLATION_AVERAGING_SOFTL1 )  {
      std::cerr << "\n Translation averaging method is invalid" << std::endl;
      return false;
    }

    use_motion_priors = cmd.used('P');
    
    return true;
  }

  std::string image_dir;
  std::string file_database = "";
  std::string output_dir = "";
  std::string k_matrix;
  std::string vid_device;
  bool up_right = false;
  std::string image_describer_method = "SIFT";
  std::string feature_preset = "";
#ifdef OPENMVG_USE_OPENMP
  int num_threads = 0;
#endif

  std::string geometric_model = "f";
  int i_user_camera_model = PINHOLE_CAMERA_RADIAL3;
  int pipeline = PIPELINE_HYBRID;
  bool group_camera_model = true;
  int gps_xyz_method = 0;
  double focal_pixels = -1.0;
  float dist_ratio = 0.8f;
  int matching_video_mode = -1;
  std::string predefined_pair_list = "";
  std::string nearest_matching_method = "AUTO";
  bool guided_matching = false;
  int imax_iteration = 2048;
  unsigned int ui_max_cache_size = 0;

  std::string s_intrinsic_refinement_options = "ADJUST_ALL";
  bool use_motion_priors = true;
  int triangulation_method = static_cast<int>(ETriangulationMethod::DEFAULT);
  int resection_method  = static_cast<int>(resection::SolverType::DEFAULT);
  int rotation_averaging_method = int (ROTATION_AVERAGING_L2);
  int translation_averaging_method = int (TRANSLATION_AVERAGING_SOFTL1);
  EINTRINSIC e_user_camera_model;
  EPairMode e_pair_mode;
  EGeometricModel e_geometric_model_to_compute;
  cameras::Intrinsic_Parameter_Type intrinsic_refinement_options;
  double focal = -1;
  double ppx = -1;
  double ppy = -1;
  std::unique_ptr<Image_describer> image_describer;
};

/// Export camera poses positions as a Vec3 vector
void GetCameraPositions(const SfM_Data & sfm_data, std::vector<Vec3> & vec_camPosition)
{
  for (const auto & view : sfm_data.GetViews())
  {
    if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get()))
    {
      const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view.second.get());
      vec_camPosition.push_back(pose.center());
    }
  }
}

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

// Find the color of the SfM_Data Landmarks/structure
bool ColorizeVideoTracks(
  const SfM_Data & sfm_data,
  std::vector<Vec3> & vec_3dPoints,
  std::vector<Vec3> & vec_tracksColor)
{
  // Colorize each track
  // Start with the most representative image
  //   and iterate to provide a color to each 3D point

  {
    C_Progress_display my_progress_bar(sfm_data.GetLandmarks().size(),
                                       std::cout,
                                       "\nCompute scene structure color\n");

    vec_tracksColor.resize(sfm_data.GetLandmarks().size());
    vec_3dPoints.resize(sfm_data.GetLandmarks().size());

    //Build a list of contiguous index for the trackIds
    std::map<IndexT, IndexT> trackIds_to_contiguousIndexes;
    IndexT cpt = 0;
    for (Landmarks::const_iterator it = sfm_data.GetLandmarks().begin();
      it != sfm_data.GetLandmarks().end(); ++it, ++cpt)
    {
      trackIds_to_contiguousIndexes[it->first] = cpt;
      vec_3dPoints[cpt] = it->second.X;
    }

    // The track list that will be colored (point removed during the process)
    std::set<IndexT> remainingTrackToColor;
    std::transform(sfm_data.GetLandmarks().begin(), sfm_data.GetLandmarks().end(),
      std::inserter(remainingTrackToColor, remainingTrackToColor.begin()),
      stl::RetrieveKey());

    while ( !remainingTrackToColor.empty() )
    {
      // Find the most representative image (for the remaining 3D points)
      //  a. Count the number of observation per view for each 3Dpoint Index
      //  b. Sort to find the most representative view index

      std::map<IndexT, IndexT> map_IndexCardinal; // ViewId, Cardinal
      for (const auto & track_to_color_it : remainingTrackToColor)
      {
        const auto trackId = track_to_color_it;
        const Observations & obs = sfm_data.GetLandmarks().at(trackId).obs;
        for (const auto & obs_it : obs)
        {
          const auto viewId = obs_it.first;
          if (map_IndexCardinal.find(viewId) == map_IndexCardinal.end())
            map_IndexCardinal[viewId] = 1;
          else
            ++map_IndexCardinal[viewId];
        }
      }

      // Find the View index that is the most represented
      std::vector<IndexT> vec_cardinal;
      std::transform(map_IndexCardinal.begin(),
        map_IndexCardinal.end(),
        std::back_inserter(vec_cardinal),
        stl::RetrieveValue());
      using namespace stl::indexed_sort;
      std::vector<sort_index_packet_descend<IndexT, IndexT>> packet_vec(vec_cardinal.size());
      sort_index_helper(packet_vec, &vec_cardinal[0], 1);

      // First image index with the most of occurence
      std::map<IndexT, IndexT>::const_iterator iterTT = map_IndexCardinal.begin();
      std::advance(iterTT, packet_vec[0].index);
      const size_t view_index = iterTT->first;
      VideoView * view = (VideoView*)sfm_data.GetViews().at(view_index).get();

      image::Image<image::RGBColor> image_rgb;
      image::Image<unsigned char> image_gray;
      
      const bool b_rgb_image = true;
      RGBColor * ptrCol = reinterpret_cast<RGBColor*>( view->frame->data[0] );
      image_rgb = Eigen::Map<Image<RGBColor>::Base>( ptrCol, view->ui_height, view->ui_width );

      // Iterate through the remaining track to color
      // - look if the current view is present to color the track
      std::set<IndexT> set_toRemove;
      for (const auto & track_to_color_it : remainingTrackToColor)
      {
        const auto trackId = track_to_color_it;
        const Observations & obs = sfm_data.GetLandmarks().at(trackId).obs;
        Observations::const_iterator it = obs.find(view_index);

        if (it != obs.end())
        {
          // Color the track
          const Vec2 & pt = it->second.x;
          const image::RGBColor color =
            b_rgb_image
            ? image_rgb(pt.y(), pt.x())
            : image::RGBColor(image_gray(pt.y(), pt.x()));

          vec_tracksColor[trackIds_to_contiguousIndexes.at(trackId)] =
            Vec3(color.r(), color.g(), color.b());
          set_toRemove.insert(trackId);
          ++my_progress_bar;
        }
      }
      // Remove colored track
      for (const auto & to_remove_it : set_toRemove)
      {
        remainingTrackToColor.erase(to_remove_it);
      }
    }
  }
  return true;
}

void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) {
    FILE *pFile;
    char szFilename[32];
    int  y;

    // Open file
    sprintf(szFilename, "frame%d.ppm", iFrame);
    pFile=fopen(szFilename, "wb");
    if(pFile==NULL)
        return;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for(y=0; y<height; y++)
        fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

    // Close file
    fclose(pFile);
}

//---------------------------------------
// Global SfM reconstruction process
//---------------------------------------
bool process_global_reconstruction(
  SfM_Data &sfm_data, Settings &settings, 
  std::shared_ptr<Features_Provider> &feats_provider,
  std::shared_ptr<Matches_Provider> &matches_provider)
{
  GlobalSfMReconstructionEngine_RelativeMotions sfmEngine(
    sfm_data,
    settings.output_dir,
    stlplus::create_filespec(settings.output_dir, "Reconstruction_Report_Global.html"));

  // Configure the features_provider & the matches_provider
  sfmEngine.SetFeaturesProvider(feats_provider.get());
  sfmEngine.SetMatchesProvider(matches_provider.get());

  // Configure reconstruction parameters
  sfmEngine.Set_Intrinsics_Refinement_Type(settings.intrinsic_refinement_options);
  sfmEngine.Set_Use_Motion_Prior(settings.use_motion_priors);

  // Configure motion averaging method
  sfmEngine.SetRotationAveragingMethod(
    ERotationAveragingMethod(settings.rotation_averaging_method));
  sfmEngine.SetTranslationAveragingMethod(
    ETranslationAveragingMethod(settings.translation_averaging_method));

  if (sfmEngine.Process())
  {
    std::cout << "...Generating SfM_Report.html" << std::endl;
    Generate_SfM_Report(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "SfMReconstruction_Report.html"));

    std::cout << "...Export SfM_Data to disk." << std::endl;

    Save(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "cloud_and_poses", ".ply"),
      ESfM_Data(ALL));

        // Compute the scene structure color
    std::vector<Vec3> vec_3dPoints, vec_tracksColor, vec_camPosition;
    if (ColorizeVideoTracks(sfmEngine.Get_SfM_Data(), vec_3dPoints, vec_tracksColor))
    {
      GetCameraPositions(sfmEngine.Get_SfM_Data(), vec_camPosition);

      // Export the SfM_Data scene in the expected format
      if (!plyHelper::exportToPly(vec_3dPoints, vec_camPosition, stlplus::create_filespec(settings.output_dir, "colorized.ply"), &vec_tracksColor))
      {
        cout << "Unable to colorize the point cloud" << endl;
        return false;
      }
    }
    else
    {
      cout << "Colorize tracks failed " << endl;
      return false;
    }

  }
  else
  {
    cout << "Failure processing the global engine!";    
    return false;
  }
  return true;
}

//---------------------------------------
// Sequential reconstruction process
//---------------------------------------
bool process_sequential_reconstruction(
  SfM_Data &sfm_data, Settings &settings,
  std::shared_ptr<Features_Provider> &feats_provider,
  std::shared_ptr<Matches_Provider> &matches_provider)
{
  SequentialSfMReconstructionEngine sfmEngine(
    sfm_data,
    settings.output_dir,
    stlplus::create_filespec(settings.output_dir, "Reconstruction_Report.html"));

  // Configure the features_provider & the matches_provider
  sfmEngine.SetFeaturesProvider(feats_provider.get());
  sfmEngine.SetMatchesProvider(matches_provider.get());

  // Configure reconstruction parameters
  sfmEngine.Set_Intrinsics_Refinement_Type(settings.intrinsic_refinement_options);
  sfmEngine.SetUnknownCameraType(EINTRINSIC(settings.i_user_camera_model));
  sfmEngine.Set_Use_Motion_Prior(settings.use_motion_priors);
  sfmEngine.SetTriangulationMethod(static_cast<ETriangulationMethod>(settings.triangulation_method));
  sfmEngine.SetResectionMethod(static_cast<resection::SolverType>(settings.resection_method));

  // Handle Initial pair parameter
  Pair initialPairIndex;
  initialPairIndex.first = 0;
  
  if (!sfmEngine.InitLandmarkTracks())
  {
    cout << "Failed to set the initial landmark tracks" << endl;
    return false;
  }

  int i = 0;
  int k = 1;
  for(;i < sfm_data.views.size();i++)
  {
    initialPairIndex.first = i;
    for(k=i+1;k<sfm_data.views.size() && k<(i+60);k++)
    {
      initialPairIndex.second = k;
      // Initial pair Essential Matrix and [R|t] estimation.
      if (sfmEngine.MakeInitialPair3D(initialPairIndex))
        goto done;
    }
  }

done:
  if(i >= sfm_data.views.size())
  {
    cout << "Unable to find a pair for the essential matrix" << endl;
    return false;
  }
  
  cout << "Initial pair is " << initialPairIndex.first << " and " << initialPairIndex.second << endl;

  //initialPairIndex.second = sfm_data.GetViews().size() - 1;
  sfmEngine.setInitialPair(initialPairIndex);

  if (sfmEngine.Process())
  {
    std::cout << "...Generating SfM_Report.html" << std::endl;
    Generate_SfM_Report(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "SfMReconstruction_Report.html"));

    std::cout << "...Export cloud and poses to disk." << std::endl;
    Save(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "cloud_and_poses", ".ply"),
      ESfM_Data(ALL));

      // Compute the scene structure color
    std::vector<Vec3> vec_3dPoints, vec_tracksColor, vec_camPosition;
    if (ColorizeVideoTracks(sfmEngine.Get_SfM_Data(), vec_3dPoints, vec_tracksColor))
    {
      GetCameraPositions(sfmEngine.Get_SfM_Data(), vec_camPosition);

      // Export the SfM_Data scene in the expected format
      if (!plyHelper::exportToPly(vec_3dPoints, vec_camPosition, stlplus::create_filespec(settings.output_dir, "colorized.ply"), &vec_tracksColor))
      {
        cout << "Unable to colorize the point cloud" << endl;
        return false;
      }
    }
    else
    {
      cout << "Unable to colorize the point cloud" << endl;
      return false;
    }
  }
  else
  {
    cout << "Failure processing the engine!";    
    return false;
  }

  return true;
}

//---------------------------------------
// Hybrid reconstruction process
//---------------------------------------
bool process_hybrid_reconstruction(
  SfM_Data &sfm_data, Settings &settings,
  std::shared_ptr<Features_Provider> &feats_provider,
  std::shared_ptr<Matches_Provider> &matches_provider)
{
  std::unique_ptr<SfMSceneInitializer> scene_initializer;
  scene_initializer.reset(new SfMSceneInitializerStellar(sfm_data,
    feats_provider.get(),
    matches_provider.get()));

  SequentialSfMReconstructionEngine2 sfmEngine(
    scene_initializer.get(),
    sfm_data,
    settings.output_dir,
    stlplus::create_filespec(settings.output_dir, "Reconstruction_Report.html"));

  // Configure the features_provider & the matches_provider
  sfmEngine.SetFeaturesProvider(feats_provider.get());
  sfmEngine.SetMatchesProvider(matches_provider.get());

  // Configure reconstruction parameters
  sfmEngine.Set_Intrinsics_Refinement_Type(settings.intrinsic_refinement_options);
  sfmEngine.SetUnknownCameraType(EINTRINSIC(settings.i_user_camera_model));
  sfmEngine.SetTriangulationMethod(static_cast<ETriangulationMethod>(settings.triangulation_method));
  sfmEngine.SetResectionMethod(static_cast<resection::SolverType>(settings.resection_method));

  if (sfmEngine.Process())
  {
    std::cout << "...Generating SfM_Report.html" << std::endl;
    Generate_SfM_Report(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "SfMReconstruction_Report.html"));

    //-- Export to disk computed scene (data & visualizable results)
    std::cout << "...Export SfM_Data to disk." << std::endl;
    //Save(sfmEngine.Get_SfM_Data(),
    //  stlplus::create_filespec(settings.output_dir, "sfm_data", ".bin"),
    //  ESfM_Data(ALL));

    Save(sfmEngine.Get_SfM_Data(),
      stlplus::create_filespec(settings.output_dir, "cloud_and_poses", ".ply"),
      ESfM_Data(ALL));

    // Compute the scene structure color
    std::vector<Vec3> vec_3dPoints, vec_tracksColor, vec_camPosition;
    if (ColorizeVideoTracks(sfmEngine.Get_SfM_Data(), vec_3dPoints, vec_tracksColor))
    {
      GetCameraPositions(sfmEngine.Get_SfM_Data(), vec_camPosition);

      // Export the SfM_Data scene in the expected format
      if (!plyHelper::exportToPly(vec_3dPoints, vec_camPosition, stlplus::create_filespec(settings.output_dir, "colorized.ply"), &vec_tracksColor))
      {
        cout << "Unable to colorize the point cloud" << endl;
        return false;
      }
    }
    else
    {
      cout << "Colorize tracks failed " << endl;
      return false;
    }
  }
  else
  {
    cout << "Failure processing the sequential engine!";
    return false;
  }
  return true;
}

//---------------------------------------
// Compute putative descriptor matches
//    - Descriptor matching (according user method choice)
//    - Keep correspondences only if NearestNeighbor ratio is ok
//---------------------------------------
bool compute_putative_descriptor(
        SfM_Data &sfm_data,
        PairWiseMatches &map_PutativesMatches,	
        const std::shared_ptr<Regions_Provider> &regions_provider, 
        const std::unique_ptr<Regions> &regions_type,
        Settings &settings)
{
  C_Progress_display progress;

  std::cout << "Use: ";
  switch (settings.e_pair_mode)
  {
    case PAIR_EXHAUSTIVE: std::cout << "exhaustive pairwise matching" << std::endl; break;
    case PAIR_CONTIGUOUS: std::cout << "sequence pairwise matching" << std::endl; break;
    case PAIR_FROM_FILE:  std::cout << "user defined pairwise matching" << std::endl; break;
  }

  // Allocate the right Matcher according the Matching requested method
  std::unique_ptr<Matcher> collectionMatcher;
  if (settings.nearest_matching_method == "AUTO")
  {
    if (regions_type->IsScalar())
    {
      std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
      collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(settings.dist_ratio));
    }
    else
    if (regions_type->IsBinary())
    {
      std::cout << "Using BRUTE_FORCE_HAMMING matcher" << std::endl;
      collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, BRUTE_FORCE_HAMMING));
    }
  }
  else
  if (settings.nearest_matching_method == "BRUTEFORCEL2")
  {
    std::cout << "Using BRUTE_FORCE_L2 matcher" << std::endl;
    collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, BRUTE_FORCE_L2));
  }
  else
  if (settings.nearest_matching_method == "BRUTEFORCEHAMMING")
  {
    std::cout << "Using BRUTE_FORCE_HAMMING matcher" << std::endl;
    collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, BRUTE_FORCE_HAMMING));
  }
  else
  if (settings.nearest_matching_method == "HNSWL2")
  {
    std::cout << "Using HNSWL2 matcher" << std::endl;
    collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, HNSW_L2));
  }
  else
  if (settings.nearest_matching_method == "ANNL2")
  {
    std::cout << "Using ANN_L2 matcher" << std::endl;
    collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, ANN_L2));
  }
  else
  if (settings.nearest_matching_method == "CASCADEHASHINGL2")
  {
    std::cout << "Using CASCADE_HASHING_L2 matcher" << std::endl;
    collectionMatcher.reset(new Matcher_Regions(settings.dist_ratio, CASCADE_HASHING_L2));
  }
  else
  if (settings.nearest_matching_method == "FASTCASCADEHASHINGL2")
  {
    std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
    collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(settings.dist_ratio));
  }
  if (!collectionMatcher)
  {
    std::cerr << "Invalid Nearest Neighbor method: " << settings.nearest_matching_method << std::endl;
    return false;
  }
  
  // Perform the matching
  {
    // From matching mode compute the pair list that have to be matched:
    Pair_Set pairs;
    switch (settings.e_pair_mode)
    {
      case PAIR_EXHAUSTIVE: pairs = exhaustivePairs(sfm_data.GetViews().size()); break;
      case PAIR_CONTIGUOUS: pairs = contiguousWithOverlap(sfm_data.GetViews().size(), settings.matching_video_mode); break;
      case PAIR_FROM_FILE:
       if (!loadPairs(sfm_data.GetViews().size(), settings.predefined_pair_list, pairs))
       {
           return false;
       }
       break;
    }
    // Photometric matching of putative pairs
    collectionMatcher->Match(regions_provider, pairs, map_PutativesMatches, &progress);
  }
  return true;
}

bool compute_geometric_matches(
        SfM_Data &sfm_data,
        PairWiseMatches &map_geometric_matches,
	const std::unique_ptr<ImageCollectionGeometricFilter> &filter_ptr,
        const PairWiseMatches &map_PutativesMatches,
	const std::shared_ptr<Regions_Provider> &regions_provider,
        Settings &settings)
{
  C_Progress_display progress;
  if (filter_ptr)
  {
    const double d_distance_ratio = 0.6;

    switch (settings.e_geometric_model_to_compute)
    {
      case HOMOGRAPHY_MATRIX:
      {
        const bool geometric_only_guided_matching = true;
        filter_ptr->Robust_model_estimation(
          GeometricFilter_HMatrix_AC(4.0, settings.imax_iteration),
          map_PutativesMatches, settings.guided_matching,
          geometric_only_guided_matching ? -1.0 : d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();
      }
      break;
      case FUNDAMENTAL_MATRIX:
      {
        filter_ptr->Robust_model_estimation(
          GeometricFilter_FMatrix_AC(4.0, settings.imax_iteration),
          map_PutativesMatches, settings.guided_matching, d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();
        std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
        new ImageCollectionGeometricFilter(&sfm_data, regions_provider));
      }
      break;
      case ESSENTIAL_MATRIX:
      {
        filter_ptr->Robust_model_estimation(
          GeometricFilter_EMatrix_AC(4.0, settings.imax_iteration),
          map_PutativesMatches, settings.guided_matching, d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();

        //-- Perform an additional check to remove pairs with poor overlap
        std::vector<PairWiseMatches::key_type> vec_toRemove;
        for (const auto & pairwisematches_it : map_geometric_matches)
        {
          const size_t putativePhotometricCount = map_PutativesMatches.find(pairwisematches_it.first)->second.size();
          const size_t putativeGeometricCount = pairwisematches_it.second.size();
          const float ratio = putativeGeometricCount / static_cast<float>(putativePhotometricCount);
          if (putativeGeometricCount < 50 || ratio < .3f)  {
            // the pair will be removed

            vec_toRemove.push_back(pairwisematches_it.first);
          }
        }
       //-- remove discarded pairs
        for (const auto & pair_to_remove_it : vec_toRemove)
        {
          map_geometric_matches.erase(pair_to_remove_it);
        }
      }
      break;
      case ESSENTIAL_MATRIX_ANGULAR:
      {
        filter_ptr->Robust_model_estimation(
          GeometricFilter_ESphericalMatrix_AC_Angular<false>(4.0, settings.imax_iteration),
          map_PutativesMatches, settings.guided_matching, d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();
      }
      break;
      case ESSENTIAL_MATRIX_ORTHO:
      {
        filter_ptr->Robust_model_estimation(
          GeometricFilter_EOMatrix_RA(2.0, settings.imax_iteration),
          map_PutativesMatches, settings.guided_matching, d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();
      }
      break;
      case ESSENTIAL_MATRIX_UPRIGHT:
      {
        filter_ptr->Robust_model_estimation(
          GeometricFilter_ESphericalMatrix_AC_Angular<true>(4.0, settings.imax_iteration),
            map_PutativesMatches, settings.guided_matching, d_distance_ratio, &progress);
        map_geometric_matches = filter_ptr->Get_geometric_matches();
      }
      break;
    }
  }
  return true;
}

void video_thread(Settings &settings)
{
  SfM_Data* sfm_data = new SfM_Data();
  double width = -1, height = -1;
  int response = 0;
  int frameFinished;
  int cnt = 0;

  // Init the media library
  avdevice_register_all();

  AVFormatContext *pFormatContext = avformat_alloc_context();
  if (!pFormatContext) {
    cout << "ERROR could not allocate memory for Format Context" << endl;
    return;
  }

  //AVInputFormat *inputFormat =av_find_input_format("v4l2");
  AVInputFormat *inputFormat;
  if (settings.vid_device.find("mp4") != std::string::npos) {
    inputFormat =av_find_input_format("mp4");
  }
  else
  {
    inputFormat =av_find_input_format("v4l2");
  }

  AVDictionary *options = NULL;
  //av_dict_set(&options, "framerate", "10", 0);

  if (avformat_open_input(&pFormatContext, settings.vid_device.c_str(), inputFormat, NULL) != 0) {
    cout << "ERROR could not open the file" << endl;
    return;
  }

  if (avformat_find_stream_info(pFormatContext,  NULL) < 0) {
    cout << "ERROR could not get the stream info" << endl;
    return;
  }

  AVCodec *pCodec = NULL;
  AVCodecParameters *pCodecParameters =  NULL;
  int video_stream_index = -1;

  for (int i = 0; i < pFormatContext->nb_streams; i++)
  {
    AVCodecParameters *pLocalCodecParameters =  NULL;
    pLocalCodecParameters = pFormatContext->streams[i]->codecpar;

    AVCodec *pLocalCodec = NULL;

    // finds the registered decoder for a codec ID
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
    pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);

    if (pLocalCodec==NULL) {
      cout << "ERROR unsupported codec!" << endl;
      // In this example if the codec is not found we just skip it
      continue;
    }

    // when the stream is a video we store its index, codec parameters and codec
    if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      if (video_stream_index == -1) {
        video_stream_index = i;
        pCodec = pLocalCodec;
        pCodecParameters = pLocalCodecParameters;
      }

    }

    // print its name, id and bitrate
    cout << "Codec " << pLocalCodec->name << endl;
  }

  if (video_stream_index == -1) {
    cout << "File does not contain a video stream! " << endl;
    return;
  }

  AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext)
  {
    cout << "failed to allocated memory for AVCodecContext" << endl;
    return;
  }

  if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0)
  {
    cout << "failed to copy codec params to codec context" << endl;
    return;
  }

  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0)
  {
    cout << "failed to open codec through avcodec_open2" << endl;
    return;
  }

  AVFrame *pFrame = av_frame_alloc();
  if (!pFrame)
  {
    cout << "failed to allocated memory for AVFrame" << endl;
    return;
  }

  AVPacket *pPacket = av_packet_alloc();
  if (!pPacket)
  {
    cout << "failed to allocated memory for AVPacket" << endl;
    return;
  }

  SwsContext *sws_ctx = sws_getContext(
                pCodecContext->width, pCodecContext->height,
                pCodecContext->pix_fmt,
                pCodecContext->width, pCodecContext->height,
                AV_PIX_FMT_RGB24, SWS_BICUBIC,
                NULL, NULL, NULL);
  Views & views = sfm_data->views;

  while (av_read_frame(pFormatContext, pPacket) >= 0)
  {
    // if it's the video stream
    if (pPacket->stream_index == video_stream_index) {

      int response = avcodec_send_packet(pCodecContext, pPacket);
      if (response < 0) {
        cout << "Error while sending a packet to the decoder " << av_err2str(response) << endl;
        goto done;
      }

      response = avcodec_receive_frame(pCodecContext, pFrame);
      if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
        continue;
      } else if (response < 0) {
        cout << "Error while receiving a frame from the decoder " << av_err2str(response) << endl;
        goto done;
      }

      if(response >= 0 && pFrame->pict_type == AV_PICTURE_TYPE_I)
      {
	cnt++;
        cout << "Frame " << views.size() << endl;
        width = pFrame->width;
        height = pFrame->height;

        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                                        width,
                                        height, 1);
        AVFrame *vFrame = av_frame_alloc();

	uint8_t* buffer = (uint8_t *) av_malloc(numBytes*sizeof(uint8_t));
        av_image_fill_arrays(vFrame->data, vFrame->linesize,
                           buffer, AV_PIX_FMT_RGB24, width, height, 1);

        sws_scale(sws_ctx, (const uint8_t * const *)pFrame->data,
                          pFrame->linesize, 0, height,
                          vFrame->data, vFrame->linesize);


        std::shared_ptr<VideoView> v;
        v = make_shared<VideoView>(vFrame,
                    views.size(), views.size(), views.size(), width, height);

        // Add the view to the sfm_container
        views[v->id_view] = v;
      }
    }
    av_packet_unref(pPacket);
  }

done:

  // pushing an item onto the queue
  vid_queue.push(sfm_data);

  // clean-up
  avformat_close_input(&pFormatContext);
  av_packet_free(&pPacket);
  av_frame_free(&pFrame);
  avcodec_free_context(&pCodecContext);
  
  // all done
  cout << "Video processing thread exiting" << endl;
  vid_done = true;
}


//
// Create the description of an input image dataset for OpenMVG toolsuite
// - Export a SfM_Data file with View & Intrinsic data
//
int main(int argc, char **argv)
{
  bool success;
  CmdLine cmd;
  Settings settings;

  settings.make_options(cmd);

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch (const std::string& s) {
      settings.output_usage(argv[0]);
      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }

  if(!settings.init(cmd))
  {
    std::cerr << "Ubable to initialize the pipeline." << std::endl;
    return EXIT_FAILURE;
  }

  settings.display(argv[0]);

  thread vid(video_thread, std::ref(settings));

  system::Timer timer;

  // Loop for batches 
  while(!vid_done || vid_queue.size() > 0)
  {
    SfM_Data* sfm_data = vid_queue.pop();
    if(sfm_data == nullptr)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));   
      continue;
    }

    Intrinsics & intrinsics = sfm_data->intrinsics;

    for (Views::const_iterator iter = sfm_data->GetViews().begin();
      iter != sfm_data->GetViews().end(); ++iter)
    {
      VideoView* view = (VideoView*)iter->second.get();
      cout << "Processing view " << view->id_view << endl;
      // Build intrinsic parameter related to the view
      std::shared_ptr<IntrinsicBase> intrinsic;
      if (settings.focal > 0 && settings.ppx > 0 && settings.ppy > 0 && view->ui_width > 0 && view->ui_height > 0)
      {
        // Create the desired camera type
        switch (settings.e_user_camera_model)
        {
          case PINHOLE_CAMERA:
            intrinsic = std::make_shared<Pinhole_Intrinsic>
              (view->ui_width, view->ui_height, settings.focal, settings.ppx, settings.ppy);
            break;
          case PINHOLE_CAMERA_RADIAL1:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K1>
              (view->ui_width, view->ui_height, settings.focal, settings.ppx, settings.ppy, 0.0); // setup no distortion as initial guess
            break;
          case PINHOLE_CAMERA_RADIAL3:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>
              (view->ui_width, view->ui_height, settings.focal, settings.ppx, settings.ppy, 0.0, 0.0, 0.0);  // setup no distortion as initial guess
            break;
          case PINHOLE_CAMERA_BROWN:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Brown_T2>
              (view->ui_width, view->ui_height, settings.focal, settings.ppx, settings.ppy, 0.0, 0.0, 0.0, 0.0, 0.0); // setup no distortion as initial guess
            break;
          case PINHOLE_CAMERA_FISHEYE:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Fisheye>
              (view->ui_width, view->ui_height, settings.focal, settings.ppx, settings.ppy, 0.0, 0.0, 0.0, 0.0); // setup no distortion as initial guess
            break;
          case CAMERA_SPHERICAL:
             intrinsic = std::make_shared<Intrinsic_Spherical>
               (view->ui_width, view->ui_height);
            break;
          default:
            std::cerr << "Error: unknown camera model: " << (int) settings.e_user_camera_model << std::endl;
          }
        }

        // Add intrinsic related to the image (if any)
        if (intrinsic == nullptr)
        {
          //Since the view have invalid intrinsic data
          // (export the view, with an invalid intrinsic field value)
          view->id_intrinsic = UndefinedIndexT;
        }
        else
        {
          // Add the defined intrinsic to the sfm_container
          intrinsics[view->id_intrinsic] = intrinsic;
        }

        RGBColor * ptrCol = reinterpret_cast<RGBColor*>( view->frame->data[0] );
        Image<RGBColor> rgb_image;
        rgb_image = Eigen::Map<Image<RGBColor>::Base>( ptrCol, view->ui_height, view->ui_width );
        Image<unsigned char> image;
        ConvertPixelType( rgb_image, &image );
        view->regions = settings.image_describer->Describe(image,nullptr);
    }

    if (settings.group_camera_model)
    {
      GroupSharedIntrinsics(*sfm_data);
    }

    Views::const_iterator iterViews = sfm_data->views.begin();
    shared_ptr<VideoView> vs = (shared_ptr<VideoView>&)iterViews->second;
    std::unique_ptr<Regions> regions_type(vs->regions->EmptyClone());
    std::shared_ptr<Regions_Provider> regions_provider = std::make_shared<Regions_Provider_Video>(sfm_data->views, regions_type);

    PairWiseMatches map_PutativesMatches;

    success = compute_putative_descriptor(
      *sfm_data, map_PutativesMatches, regions_provider, regions_type, settings);

    if(!success)
    {
      cout << "Failure to calculate the putative descriptor." << endl;
      return EXIT_FAILURE;
    }

    std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
      new ImageCollectionGeometricFilter(sfm_data, regions_provider));

    PairWiseMatches map_geometric_matches;

    success = compute_geometric_matches(
      *sfm_data, map_geometric_matches, filter_ptr, map_PutativesMatches,regions_provider, settings );
    if(!success)
    {
      cout << "Failure to calculate the geometric matching." << endl;
      return EXIT_FAILURE;
    }

    // Features reading
    std::shared_ptr<Features_Provider> feats_provider = std::make_shared<Features_Provider_Video>(*sfm_data, regions_provider);

    // Matches reading
    std::shared_ptr<Matches_Provider> matches_provider = std::make_shared<Matches_Provider_Video>(*sfm_data, map_geometric_matches);

    switch(settings.pipeline)
    {
      case PIPELINE_GLOBAL:
        success = process_global_reconstruction(*sfm_data, settings, feats_provider, matches_provider);
        break;
      case PIPELINE_SEQUENTIAL:
        success = process_sequential_reconstruction(*sfm_data, settings, feats_provider, matches_provider);
      case PIPELINE_HYBRID:
        success = process_hybrid_reconstruction(*sfm_data, settings, feats_provider, matches_provider);
        break;
    }

    if(success)
      cout << "Successfully processed the pipeline." << endl;
    else
      cout << "Failure to process the pipeline." << endl;

    for (int i = 0; i < static_cast<int>(sfm_data->views.size()); ++i)
    {
      Views::const_iterator iterViews = sfm_data->views.begin();
      std::advance(iterViews, i);
      VideoView * view = (VideoView*)iterViews->second.get();

      av_frame_free(&view->frame);
    }

    delete sfm_data;
  }

  std::cout << std::endl << "******* Total time takens in (s): " << timer.elapsed() << std::endl;
  
  cout << "Returning." << endl;
  return 0;
}
