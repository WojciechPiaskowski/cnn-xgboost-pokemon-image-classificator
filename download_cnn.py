import gdown


def download_cnn():
    url = 'https://drive.google.com/file/d/1GdLZrCJVaW0sdx6X3WfNvyhfv_h4br6U/view?usp=share_link'
    file_id = url.split('/')[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    gdown.download(prefix+file_id, output='models/cnn/cnn.h5')