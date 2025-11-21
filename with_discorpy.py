import numpy as np
import matplotlib.pyplot as plt
import discorpy.losa.loadersaver as losa
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

# --- User parameters ---
file_path = "table.jpeg"  # Input table image
output_base = "./"        # Output folder
num_coef = 4              # Number of polynomial coefficients

# --- Load image ---
mat0 = losa.load_image(file_path)
(height, width) = mat0.shape

# --- Normalize background ---
mat1 = prep.normalization_fft(mat0, sigma=20)

# --- Segment dots ---
threshold = prep.calculate_threshold(mat1, bgr="bright", snr=1.5)
mat1 = prep.binarization(mat1, thres=threshold)
losa.save_image(output_base + "segmented_dots.jpg", mat1)

# --- Calculate dot size and distance ---
(dot_size, dot_dist) = prep.calc_size_distance(mat1)

# --- Calculate slopes ---
hor_slope = prep.calc_hor_slope(mat1)
ver_slope = prep.calc_ver_slope(mat1)
print("Horizontal slope: {0}, Vertical slope: {1}".format(hor_slope, ver_slope))

# --- Group dots into horizontal and vertical lines ---
list_hor_lines0 = prep.group_dots_hor_lines(mat1, hor_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=2,
                                            accepted_ratio=0.6)
list_ver_lines0 = prep.group_dots_ver_lines(mat1, ver_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=2,
                                            accepted_ratio=0.6)

# --- Save line plots for checking ---
losa.save_plot_image(output_base + "horizontal_lines.png", list_hor_lines0, height, width)
losa.save_plot_image(output_base + "vertical_lines.png", list_ver_lines0, height, width)

# --- Residuals before correction ---
list_hor_data = post.calc_residual_hor(list_hor_lines0, 0.0, 0.0)
list_ver_data = post.calc_residual_ver(list_ver_lines0, 0.0, 0.0)
losa.save_residual_plot(output_base + "hor_residual_before_correction.png", list_hor_data, height, width)
losa.save_residual_plot(output_base + "ver_residual_before_correction.png", list_ver_data, height, width)

# --- Correct perspective distortion ---
list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(list_hor_lines0, list_ver_lines0, perspective=True)

# --- Determine radial distortion coefficients ---
(xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines1, list_ver_lines1)
list_fact = proc.calc_coef_backward(list_hor_lines1, list_ver_lines1, xcenter, ycenter, num_coef)
losa.save_metadata_txt(output_base + "coefficients_radial_distortion.txt", xcenter, ycenter, list_fact)
print("Radial distortion center: X={0}, Y={1}".format(xcenter, ycenter))
print("Radial distortion coefficients:", list_fact)

# --- Unwarp lines using backward model ---
list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(list_hor_lines0, list_ver_lines0, perspective=False)
list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter, list_fact)
list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter, list_fact)

# --- Check residuals after radial correction ---
list_hor_data = post.calc_residual_hor(list_uhor_lines, xcenter, ycenter)
list_ver_data = post.calc_residual_ver(list_uver_lines, xcenter, ycenter)
losa.save_residual_plot(output_base + "hor_residual_after_correction.png", list_hor_data, height, width)
losa.save_residual_plot(output_base + "ver_residual_after_correction.png", list_ver_data, height, width)

# --- Unwarp image (radial correction) ---
mat_rad_corr = post.unwarp_image_backward(mat0, xcenter, ycenter, list_fact)
losa.save_image(output_base + "image_radial_corrected.jpg", mat_rad_corr)
losa.save_image(output_base + "radial_difference.jpg", mat_rad_corr - mat0)

# --- Correct perspective ---
source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,
                                                                              equal_dist=True, scale="mean",
                                                                              optimizing=False)
pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")
image_pers_corr = post.correct_perspective_image(mat_rad_corr, pers_coef)
np.savetxt(output_base + "perspective_coefficients.txt", np.transpose([pers_coef]))

# --- Save final corrected image ---
losa.save_image(output_base + "table_corrected.jpeg", image_pers_corr)
losa.save_image(output_base + "perspective_difference.jpg", image_pers_corr - mat_rad_corr)

print("Distortion correction completed. Final image saved as table_corrected.jpeg")
