#ifndef OPENMVG_SFM_SFM_VIDEO_VIEW_HPP
#define OPENMVG_SFM_SFM_VIDEO_VIEW_HPP

#include "openMVG/sfm/sfm_view.hpp"
#include <string>
#include <vector>

namespace openMVG {
namespace sfm {

struct View;

struct VideoView : public View {
public:

  VideoView() : View() { }

  // Constructor (use unique index for the view_id)
  VideoView(
    AVFrame *frame_id,
    IndexT view_id = UndefinedIndexT,
    IndexT intrinsic_id = UndefinedIndexT,
    IndexT pose_id = UndefinedIndexT,
    IndexT width = UndefinedIndexT, IndexT height = UndefinedIndexT) :
	  View("",view_id,intrinsic_id,pose_id,width,height),
          frame(frame_id)
  {
  }

  ~VideoView() override = default;

  template <class Archive>
  void save( Archive & ar ) const;

  template <class Archive>
  void load( Archive & ar );

  // image data in memory
  AVFrame* frame;
  //std::unique_ptr<openMVG::features::Regions> regions;
  std::unique_ptr<openMVG::features::Regions> regions;
};
}
}

#endif
