def getVideoData(video_input):

    VIDEO_DATA = {}
    VIDEO_DATA['project_video.mp4'] = {}
    VIDEO_DATA['challenge_video.mp4'] = {}
    VIDEO_DATA['harder_challenge_video.mp4'] = {}


    # PROJECT VIDEO
    VIDEO_DATA['project_video.mp4']['ROI_POINTS'] = [(0.13516, 0.94444), (0.43906, 0.59028), (0.57266, 0.59028), (1.0, 0.94444)]
    VIDEO_DATA['project_video.mp4']['TRANSFORM_SRC_POINTS'] = [(0.23828, 0.98611), (0.46563, 0.6375), (0.56406, 0.6375), (0.95312, 0.98611)]
    VIDEO_DATA['project_video.mp4']['BRIGHTNESS_THRESHOLD'] = 100

    VIDEO_DATA['project_video.mp4']['L_THRESHOLD_LC']  = (118, 255)
    VIDEO_DATA['project_video.mp4']['B_THRESHOLD_LC']  = (159, 255)
    VIDEO_DATA['project_video.mp4']['L_THRESHOLD']     = (199, 255)
    VIDEO_DATA['project_video.mp4']['B_THRESHOLD']     = (159, 255)


    # CHALLENGE VIDEO
    VIDEO_DATA['challenge_video.mp4']['ROI_POINTS'] = [(0.15625, 0.92917), (0.51016, 0.63194), (0.57344, 0.63194), (0.92266, 0.92917)]
    VIDEO_DATA['challenge_video.mp4']['TRANSFORM_SRC_POINTS'] = [(0.18672, 1.0), (0.47813, 0.65972), (0.57422, 0.65972), (0.89766, 1.0)]
    VIDEO_DATA['challenge_video.mp4']['BRIGHTNESS_THRESHOLD'] = 75

    VIDEO_DATA['challenge_video.mp4']['L_THRESHOLD_LC'] = (176, 255)
    VIDEO_DATA['challenge_video.mp4']['B_THRESHOLD_LC'] = (131, 255)
    VIDEO_DATA['challenge_video.mp4']['L_THRESHOLD']    = (175, 255)
    VIDEO_DATA['challenge_video.mp4']['B_THRESHOLD']    = (144, 255)

    # HARDER_CHALLENGE VIDEO
    VIDEO_DATA['harder_challenge_video.mp4']['TRANSFORM_SRC_POINTS'] = [(0.0, 0.91528), (0.27031, 0.73472), (0.73984, 0.73472), (1.0, 0.91528)]
    VIDEO_DATA['harder_challenge_video.mp4']['ROI_POINTS'] = [(0.12656, 0.93333), (0.29453, 0.60278), (1.0, 0.60278), (1.0, 0.93333)]
    VIDEO_DATA['harder_challenge_video.mp4']['BRIGHTNESS_THRESHOLD'] = 100
    VIDEO_DATA['harder_challenge_video.mp4']['L_THRESHOLD_LC'] = (125, 255)
    VIDEO_DATA['harder_challenge_video.mp4']['B_THRESHOLD_LC'] = (138, 255)
    VIDEO_DATA['harder_challenge_video.mp4']['L_THRESHOLD']    = (224, 255)
    VIDEO_DATA['harder_challenge_video.mp4']['B_THRESHOLD']    = (172, 255)

    #
    # if the video data is not available use
    # 'project_video.mp4' as default
    #
    if video_input in VIDEO_DATA:
        return VIDEO_DATA[video_input]
    else:
        return VIDEO_DATA['project_video.mp4']
