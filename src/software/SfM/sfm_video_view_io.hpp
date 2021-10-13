#ifndef OPENMVG_SFM_SFM_VIDEO_VIEW_IO_HPP
#define OPENMVG_SFM_SFM_VIDEO_VIEW_IO_HPP

#include "sfm_video_view.hpp"

#include <cereal/types/polymorphic.hpp>

template <class Archive>
void openMVG::sfm::VideoView::save( Archive & ar ) const
{
  View::save(ar);
}

template <class Archive>
void openMVG::sfm::VideoView::load( Archive & ar )
{
  View::load(ar);
}

CEREAL_REGISTER_TYPE_WITH_NAME( openMVG::sfm::VideoView, "VideoView" );
CEREAL_REGISTER_POLYMORPHIC_RELATION(openMVG::sfm::View, openMVG::sfm::VideoView);

#endif
