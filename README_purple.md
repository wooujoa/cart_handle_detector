# Cart Handle Detector - Purple Feature Version

이 문서는 **Purple + Green feature 버전만** 사용하기 위한 실행 명령어를 정리한 것이다.  
Green-only 버전은 더 이상 사용하지 않는다.

---

## 1. Feature 배치

현재 카트 손잡이 feature 배치는 다음과 같다.

```text
0.0  ~ 14.5 cm : yellow
14.5 ~ 17.5 cm : purple direction marker
17.5 ~ 25.7 cm : yellow
25.7 ~ 28.7 cm : green center marker
28.7 ~ 54.5 cm : yellow
```

중심값은 다음과 같다.

```text
purple center = 16.0 cm = 0.160 m
green center  = 27.2 cm = 0.272 m
marker gap    = 11.2 cm
```

---

## 2. purple_feature_detect 실행

```bash
ros2 run cart_handle_detector purple_feature_detect --ros-args -p color_topic:=/zedm/zed_node/rgb/image_rect_color -p publish_debug_image:=true -p debug_log:=true -p profile:=true -p purple_h_min:=122 -p purple_h_max:=179 -p purple_s_min:=20 -p purple_s_max:=168 -p purple_v_min:=87 -p purple_v_max:=207 -p open_kernel:=1 -p close_kernel:=0 -p handle_length_cm:=54.5 -p marker_gap_cm:=11.2 -p center_green_center_cm:=27.2 -p purple_width_cm:=3.0 -p green_width_cm:=3.0 -p min_pair_dist_px:=18.0 -p max_pair_dist_px:=260.0 -p endpoint_margin_px:=80.0 -p pattern_corridor_half_width_px:=28.0 -p min_yellow_total_pixels:=70 -p min_yellow_left_pixels:=15 -p min_yellow_middle_pixels:=10 -p min_yellow_right_pixels:=20 -p min_purple_pixels:=6 -p min_green_pixels:=6 -p min_lr_yellow_balance:=0.15
```

---

## 3. cart_detect 실행

```bash
ros2 run cart_handle_detector cart_detect --ros-args -p camera_info_topic:=/zedm/zed_node/rgb/camera_info -p handle_length_m:=0.545 -p end_green_center_m:=0.160 -p center_green_center_m:=0.272 -p handle_z_min_m:=1.010 -p handle_z_max_m:=1.010 -p handle_z_step_m:=0.005 -p use_cylinder_centerline_correction:=true -p handle_radius_m:=0.015 -p perp_offset_search_px:=45.0 -p perp_offset_step_px:=2.0 -p max_green_gap_error_m:=0.055 -p max_handle_length_error_m:=0.160 -p min_axis_consistency:=0.55 -p basket_side:=left -p standoff_m:=0.45 -p publish_forward_offset_m:=0.07 -p publish_yaw_offset_deg:=3.0 -p enable_temporal_gate:=true -p max_center_jump_m:=0.10 -p max_goal_jump_m:=0.15 -p max_yaw_jump_deg:=18.0 -p pending_accept_count:=4 -p pending_similarity_center_m:=0.04 -p pending_similarity_goal_m:=0.06 -p pending_similarity_yaw_deg:=8.0 -p hold_previous_on_reject:=true -p max_hold_sec:=1.0 -p force_accept_after_rejects:=10 -p enable_smoothing:=true -p xy_alpha:=0.55 -p yaw_alpha:=0.55 -p publish_markers:=true -p debug:=true -p profile:=true
```

---

## 4. 현재 적용된 publish offset

`cart_detect`에는 최종 publish 직전에 다음 offset이 적용된다.

```text
publish_forward_offset_m = 0.07
publish_yaw_offset_deg   = 3.0
```

의미는 다음과 같다.

```text
base_link 기준 +x 방향으로 7 cm 이동
base_link 기준 반시계 방향으로 +3 deg yaw offset
```

---

## 5. Temporal Gate 설정

`cart_detect` 내부 temporal gate는 1-frame 튐을 막기 위해 사용한다.

```text
max_center_jump_m          = 0.10
max_goal_jump_m            = 0.15
max_yaw_jump_deg           = 18.0
pending_accept_count       = 4
pending_similarity_center_m = 0.04
pending_similarity_goal_m   = 0.06
pending_similarity_yaw_deg  = 8.0
max_hold_sec               = 1.0
force_accept_after_rejects = 10
```

동작 방식:

```text
정상 변화:
  새 pose accept 후 smoothing

1-frame 튐:
  reject 후 이전 accepted pose publish

비슷한 튄 값이 연속으로 들어옴:
  pending_accept_count 횟수만큼 누적되면 새 pose로 인정
```

---

## 6. Debug 확인

### Feature debug image

RViz Image display에서 아래 topic을 확인한다.

```text
/cart_handle/debug_image_zed
```

### 최종 pose 확인

```bash
ros2 topic hz /cart_handle/goal_pose_base --window 50
ros2 topic echo --once /cart_handle/goal_pose_base
```

### Feature topic 확인

```bash
ros2 topic hz /cart_handle/end_green_px_zed --window 50
ros2 topic hz /cart_handle/center_green_px_zed --window 50
```

주의: topic 이름은 기존 호환성을 위해 `end_green_px_zed`를 유지하지만, Purple 버전에서는 이 topic에 **purple direction marker pixel**이 publish된다.

---

## 7. 보라색 HSV 현재값

```text
purple_h_min = 122
purple_h_max = 179
purple_s_min = 20
purple_s_max = 168
purple_v_min = 87
purple_v_max = 207
```

보라색 후보가 너무 많이 잡히면 우선 `purple_s_min`을 높이는 방향으로 조정한다.

예시:

```text
purple_s_min = 50 ~ 80
```
