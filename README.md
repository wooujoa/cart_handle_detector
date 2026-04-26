# Cart Handle Detector 실행 명령어 정리

이 문서는 카트 손잡이 feature 구성에 따라 `feature_detect`, `cart_detect`를 실행하는 명령어를 정리한 것.

---

## 1. Green Feature 버전

### Feature 패턴

```text
0.0 ~ 3.0 cm      : green direction marker
3.0 ~ 25.7 cm     : yellow handle
25.7 ~ 28.7 cm    : green center marker
28.7 ~ 54.5 cm    : yellow handle
```

### feature_detect 실행

```bash
ros2 run cart_handle_detector feature_detect --ros-args \
  -p color_topic:=/zedm/zed_node/rgb/image_rect_color \
  -p publish_debug_image:=true \
  -p debug_log:=true \
  -p profile:=true \
  -p end_green_start_cm:=0.0 \
  -p end_green_end_cm:=3.0 \
  -p center_green_start_cm:=25.7 \
  -p center_green_end_cm:=28.7 \
  -p endpoint_margin_px:=25.0 \
  -p end_green_max_perp_px:=18.0 \
  -p min_yellow_balance:=0.55
```

잘 안 잡히는 경우 완화:

```bash
ros2 run cart_handle_detector feature_detect --ros-args \
  -p color_topic:=/zedm/zed_node/rgb/image_rect_color \
  -p publish_debug_image:=true \
  -p debug_log:=true \
  -p profile:=true \
  -p end_green_start_cm:=0.0 \
  -p end_green_end_cm:=3.0 \
  -p center_green_start_cm:=25.7 \
  -p center_green_end_cm:=28.7 \
  -p endpoint_margin_px:=35.0 \
  -p end_green_max_perp_px:=24.0 \
  -p min_yellow_balance:=0.50
```

### cart_detect 실행

```bash
ros2 run cart_handle_detector cart_detect --ros-args \
  -p camera_info_topic:=/zedm/zed_node/rgb/camera_info \
  -p handle_length_m:=0.545 \
  -p end_green_center_m:=0.015 \
  -p center_green_center_m:=0.272 \
  -p handle_z_min_m:=1.010 \
  -p handle_z_max_m:=1.010 \
  -p handle_z_step_m:=0.005 \
  -p use_cylinder_centerline_correction:=true \
  -p handle_radius_m:=0.015 \
  -p perp_offset_search_px:=35.0 \
  -p perp_offset_step_px:=2.0 \
  -p max_green_gap_error_m:=0.025 \
  -p max_handle_length_error_m:=0.090 \
  -p basket_side:=left \
  -p standoff_m:=0.45 \
  -p publish_forward_offset_m:=0.02 \
  -p publish_markers:=true \
  -p debug:=true \
  -p profile:=true
```

---

## 2. Purple + Green Feature 버전

### Feature 패턴

```text
purple direction marker
green center marker

purple ↔ green 중심 거리 ≈ 7.5 cm
center green 중심 = 27.2 cm
purple 중심 = 27.2 - 7.5 = 19.7 cm
```

### feature_detect 실행

```bash
ros2 run cart_handle_detector feature_detect --ros-args \
  -p color_topic:=/zedm/zed_node/rgb/image_rect_color \
  -p publish_debug_image:=true \
  -p debug_log:=true \
  -p profile:=true \
  -p purple_h_min:=122 \
  -p purple_h_max:=179 \
  -p purple_s_min:=20 \
  -p purple_s_max:=168 \
  -p purple_v_min:=87 \
  -p purple_v_max:=207 \
  -p open_kernel:=1 \
  -p close_kernel:=0 \
  -p marker_gap_cm:=7.5 \
  -p center_green_center_cm:=27.2 \
  -p purple_width_cm:=3.0 \
  -p green_width_cm:=3.0 \
  -p min_pair_dist_px:=15.0 \
  -p max_pair_dist_px:=180.0 \
  -p endpoint_margin_px:=45.0 \
  -p min_yellow_middle_pixels:=4 \
  -p min_lr_yellow_balance:=0.25
```

### cart_detect 실행

```bash
ros2 run cart_handle_detector cart_detect --ros-args \
  -p camera_info_topic:=/zedm/zed_node/rgb/camera_info \
  -p handle_length_m:=0.545 \
  -p end_green_center_m:=0.197 \
  -p center_green_center_m:=0.272 \
  -p handle_z_min_m:=1.010 \
  -p handle_z_max_m:=1.010 \
  -p handle_z_step_m:=0.005 \
  -p use_cylinder_centerline_correction:=true \
  -p handle_radius_m:=0.015 \
  -p perp_offset_search_px:=35.0 \
  -p perp_offset_step_px:=2.0 \
  -p max_green_gap_error_m:=0.025 \
  -p max_handle_length_error_m:=0.090 \
  -p basket_side:=left \
  -p standoff_m:=0.45 \
  -p publish_forward_offset_m:=0.02 \
  -p publish_markers:=true \
  -p debug:=true \
  -p profile:=true
```

---

## 3. 주요 파라미터 요약

### Green 버전

```text
feature_detect:
  end_green_start_cm      = 0.0
  end_green_end_cm        = 3.0
  center_green_start_cm   = 25.7
  center_green_end_cm     = 28.7

cart_detect:
  end_green_center_m      = 0.015
  center_green_center_m   = 0.272
```

### Purple + Green 버전

```text
feature_detect:
  marker_gap_cm           = 7.5
  center_green_center_cm  = 27.2

cart_detect:
  end_green_center_m      = 0.197
  center_green_center_m   = 0.272
```

보라색 marker와 green marker의 실제 중심 거리가 바뀌면 아래 식으로 `cart_detect` 값을 수정한다.

```text
end_green_center_m = (27.2 - marker_gap_cm) / 100
```

예시:

```text
marker_gap_cm = 8.0
end_green_center_m = (27.2 - 8.0) / 100 = 0.192
```

---

## 4. Debug Topic

RViz에서 feature 검출 상태를 확인할 때는 아래 topic을 Image display로 본다.

```text
/cart_handle/debug_image_zed
```

최종 pose는 아래 topic으로 확인한다.

```bash
ros2 topic hz /cart_handle/goal_pose_base --window 50
ros2 topic echo --once /cart_handle/goal_pose_base
```

