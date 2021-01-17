from pymediainfo import MediaInfo
import subprocess

VIDEO_EXTENSION = ['rm', 'webm', 'mpg', 'mp2', 'mpeg', 'mpe', 'asf',
                   'mpv', 'ogg', 'mp4', 'm4p', 'm4v', 'avi',
                   'wmv', 'mov', 'qt', 'flv', 'swf', 'avchd']


def parse_video(video):
    meta = dict()
    meta['path'] = video
    code = True
    try:
        media_info = MediaInfo.parse(video)
        for track in media_info.tracks:
            if track.track_type == 'General':
                meta['file_name'] = track.file_name + '.' + track.file_extension
                meta['file_extension'] = track.file_extension
                meta['format'] = track.format
                meta['duration'] = track.duration
                meta['frame_count'] = track.frame_count
                meta['frame_rate'] = track.frame_rate
            elif track.track_type == 'Video':
                meta['width'] = int(track.width)
                meta['height'] = int(track.height)
                meta['rotation'] = float(track.rotation) if track.rotation is not None else 0.
                meta['codec'] = track.codec
    except Exception as e:
        code = False

    return code, meta


def decode_video(video, frame_dir, decode_rate):
    cmd = ['ffmpeg',
           '-i', video,
           '-vsync', '2',
           '-map', '0:v:0',
           '-q:v', '0',
           '-vf', f'fps={decode_rate}',
           '-f', 'image2',
           f'{frame_dir}/%6d.jpg']

    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _ = p.communicate()
    code = True if p.returncode != 1 else False
    return code
