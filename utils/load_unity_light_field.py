"""
Load Unity light field data (from TMNH codebase)

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Article: 
S. Choi, C. Jang, D. Lanman, G. Wetzstein, 
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025
"""

import logging
import os
import math
import json
import torch
import imageio

def load_unity_light_field(
    datapath,
    eyepieceFocalLength=None,
    frameNum=None,
    flipLFOutput=False,
    loadOnlyCentralView=False,
    channel=1,
    lf_params=None,
    dev=torch.device('cuda')
):
    # json calibration file name
    json_fname = open(os.path.join(datapath, 'cameras.json'))
    json_data = json.load(json_fname)
    json_fname.close()

    # near and far clipping planes
    zNear = json_data['NearClip']
    zFar = json_data['FarClip']

    # height and width of viewport plane
    h = json_data['ViewportHeight']
    w = json_data['ViewportWidth']

    # SLM unit scaling and indices
    if lf_params is not None:
        slmPitch = lf_params['feature_size'][0]
        stride_y = 1
        stride_x = 1
        start_y = lf_params['lf_start_idx'][0]
        end_y = lf_params['lf_end_idx'][0] + 1
        start_x = lf_params['lf_start_idx'][1]
        end_x = lf_params['lf_end_idx'][1] + 1
    else:
        slmPitch = 6.4e-6

    imageResolution = [json_data['PixelHeight'], json_data['PixelWidth']]
    imageWidth = slmPitch * imageResolution[1]
    if eyepieceFocalLength is not None:
        # scale imageWidth by magnification
        eyepieceVirtualImageDist = json_data['CameraDistance'] - eyepieceFocalLength
        eyepieceHologramDist = 1 / (1 / eyepieceFocalLength + 1 / eyepieceVirtualImageDist)
        magnification = eyepieceFocalLength / (eyepieceFocalLength - eyepieceHologramDist)
        imageWidth *= magnification

    unitScale = imageWidth / json_data['ViewportWidth']

    h *= unitScale
    w *= unitScale
    zNear *= unitScale
    zFar *= unitScale

    # get a grid for x and y coords in window coordinates
    xx_win, yy_win = torch.meshgrid(
        torch.linspace(0, imageResolution[1], imageResolution[1], device=dev),
        torch.linspace(imageResolution[0], 0, imageResolution[0], device=dev),
        indexing='ij'
    )

    xx_win = torch.transpose(xx_win, 0, 1)
    yy_win = torch.transpose(yy_win, 0, 1)

    # calculate pixel positions given depth
    xx_ndc = xx_win / imageResolution[1] - 1 / 2
    yy_ndc = yy_win / imageResolution[0] - 1 / 2

    if loadOnlyCentralView:
        # specify coordinates of the center view
        centerYView = math.floor(json_data['CameraRows'] / 2)
        centerXView = math.floor(json_data['CameraColumns'] / 2)
    else:
        # allocate memory for light field and depth
        if lf_params['num_views_to_load'] is None:
            light_field = torch.zeros(
                json_data['CameraRows'],
                json_data['CameraColumns'],
                *imageResolution,
                device=dev
            )
        else:
            light_field = torch.zeros(
                *lf_params['num_views_to_load'],
                *imageResolution, device=dev
            )
        depth = torch.zeros_like(light_field)

    camy_list = list(range(start_y, end_y, stride_y))
    camy_list.reverse()
    camx_list = list(range(start_x, end_x, stride_x))

    import logging
    logging.info(f"DataLoader: Loading {len(camy_list)} by {len(camx_list)} light field views from {datapath} ...")

    for idx_y, camy in enumerate(camy_list):
        # skip views if loading only central view
        if loadOnlyCentralView and camy != centerYView:
            continue

        for idx_x, camx in enumerate(camx_list):
            # skip views if loading only central view
            if loadOnlyCentralView and camx != centerXView:
                continue

            # camera index, flip y coordinate
            camidx = (camy - 1) * json_data['CameraColumns'] + (camx - 1)

            # camera position relative to central view
            campos = json_data['Cameras'][camidx]['parameters']['localPosition']
            campos_x = unitScale * campos['x']
            campos_y = unitScale * campos['y']

            # load depth map and light field view
            cam_key = json_data["Cameras"][camidx]["key"]
            if eyepieceFocalLength is None or frameNum is None:
                image_file_path = os.path.join(datapath, f'{cam_key}_rgbd.png')
            else:
                image_file_path = os.path.join(datapath, f'{cam_key}_rgbd_{frameNum:04d}.png')

            I = imageio.imread(image_file_path)

            # Get channel and depth - always float32 and scaled to [0,1]
            if len(I.shape) == 3 and I.shape[2] == 4:
                D = I[..., 3]
                I_rgb = I[..., :3]
            else:
                D = I[..., -1]
                I_rgb = I[..., :-1] if I.shape[-1] > 1 else I

            I_ch = I_rgb[..., channel] if I_rgb.shape[-1] > channel else I_rgb[..., 0]
            I_ch = torch.tensor(I_ch, dtype=torch.float32, device=dev) / 255.0
            D = torch.tensor(D, dtype=torch.float32, device=dev) / 255.0

            # convert to normalized double precision floating point values
            D = 1.0 / (D * (1.0 / zNear - 1.0 / zFar) + 1.0 / zFar)

            # get/reset zero disparity plane
            zero_disp_plane = json_data['CameraDistance']
            zero_disp_plane = unitScale * zero_disp_plane

            # target position on SLM / viewport / zero_disparity_plane for each pixel
            xx_slm = xx_ndc * w
            yy_slm = yy_ndc * h

            # account for camera position's depth-dependent shift
            x_offset = (zero_disp_plane - D) / zero_disp_plane * campos_x
            y_offset = (zero_disp_plane - D) / zero_disp_plane * campos_y

            # point cloud relative to central camera position
            xx_metric = xx_ndc * w * D / zero_disp_plane + x_offset
            yy_metric = yy_ndc * h * D / zero_disp_plane + y_offset

            # use focal length to convert depth to be relative to hologram plane
            # (which is assumed to be the zero disparity plane)
            if eyepieceFocalLength is not None:
                virtualImageDist = D - eyepieceFocalLength
                imageDist = 1.0 / (1.0 / eyepieceFocalLength + 1.0 / virtualImageDist)
                imageMag = eyepieceFocalLength / (eyepieceFocalLength - imageDist)
                virtualZeroDisp = zero_disp_plane - eyepieceFocalLength

                zero_disp_plane = 1.0 / (1.0 / eyepieceFocalLength + 1.0 / virtualZeroDisp)
                zeroDispMag = eyepieceFocalLength / (eyepieceFocalLength - zero_disp_plane)

                xx_metric = xx_metric / imageMag
                yy_metric = yy_metric / imageMag
                xx_slm = xx_slm / zeroDispMag
                yy_slm = yy_slm / zeroDispMag
                D = imageDist

            # positions relative to corresponding SLM pixel
            xx_dist = xx_slm - xx_metric
            yy_dist = yy_slm - yy_metric
            zz_dist = zero_disp_plane - D

            # distance from pixel to corresponding SLM pixel
            abs_dist = torch.sqrt(xx_dist ** 2 + yy_dist ** 2 + zz_dist ** 2)
            # sign for which side of slm
            metric_dist = abs_dist * zz_dist / torch.abs(zz_dist)

            if loadOnlyCentralView:
                light_field = I_ch
                depth = metric_dist
            else:
                light_field[idx_y, idx_x, ...] = I_ch
                depth[idx_y, idx_x, :, :] = metric_dist

    def flip(x, dim):
        return torch.flip(x, dims=[dim])

    if flipLFOutput and not loadOnlyCentralView:
        light_field = flip(light_field, 1)
        light_field = flip(light_field, 2)
        depth = flip(depth, 1)
        depth = flip(depth, 2)
        depth = -depth

    return light_field, depth
