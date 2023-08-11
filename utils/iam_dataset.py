import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize, centered

class IAMDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms=None):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = 'IAM'
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.form_file = '{}/{}/ascii/forms.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/{}/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = '{}/{}/ascii/words.txt'.format(self.basefolder, self.setname)
        self.form_path = '{}/{}/forms'.format(self.basefolder, self.setname)
        self.word_path = '{}/{}/words'.format(self.basefolder, self.setname)
        self.line_path = '{}/{}/lines'.format(self.basefolder, self.setname)
        self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:

        def gather_iam_info(self, level='word'):

            if subset == 'train':
                valid_set = np.loadtxt(self.trainset_file, dtype=str)
            elif subset == 'val':
                valid_set = np.loadtxt(self.valset_file, dtype=str)
            elif subset == 'test':
                valid_set = np.loadtxt(self.testset_file, dtype=str)
            else:
                raise ValueError


            if level == 'word':
                gtfile= self.word_file
                root_path = self.word_path
            elif level == 'fword':
                # extract words with context!!!
                gtfile = self.word_file
                root_path = self.form_path
                valid_set = ['-'.join(l.split('-')[:-1]) for l in valid_set]
            elif level == 'line':
                gtfile = self.line_file
                root_path = self.line_path
            elif level == 'form':
                gtfile = self.form_file
                gtextra = self.word_file
                root_path = self.form_path
                valid_set = ['-'.join(l.split('-')[:-1]) for l in valid_set]
            else:
                raise ValueError

            if level == 'form':
                form_dict = {}
                for line in open(gtextra):
                    if not line.startswith("#"):
                        info = line.strip().split()
                        name = info[0]
                        name_parts = name.split('-')
                        form_name = '-'.join(name_parts[:2])

                        if (form_name not in valid_set) or (info[1] != 'ok'):
                            continue

                        bbox = [int(info[3]), int(info[4]), int(info[5]), int(info[6])]

                        if bbox[2] < 8 and bbox[3] < 8:
                            continue

                        transcr = ' '.join(info[8:])

                        if form_name in form_dict:
                            form_dict[form_name] = form_dict[form_name] + [(bbox, transcr)]
                        else:
                            form_dict[form_name] = [(bbox, transcr)]

            gt = []
            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]

                    name_parts = name.split('-')
                    pathlist = [root_path] + ['-'.join(name_parts[:i + 1]) for i in range(len(name_parts))]
                    if level == 'word':
                        tname = pathlist[-2]
                        del pathlist[-2]
                    elif level == 'fword':
                        pathlist = pathlist[:-2]
                        tname = pathlist[-1]
                        del pathlist[-2]
                    elif level == 'line':
                        tname = pathlist[-1]
                    elif level == 'form':
                        tname = pathlist[-1]
                        del pathlist[-2]

                    if (tname not in valid_set):
                        continue

                    if 'word' in level:
                        if (info[1] != 'ok') or (tname not in valid_set):
                            continue

                    img_path = '/'.join(pathlist)

                    if level == 'form':
                        transcr = ''
                    else:
                        transcr = ' '.join(info[8:])

                    if level == 'fword':
                        bbox = [int(info[3]), int(info[4]), int(info[5]), int(info[6])]
                    elif level == 'form':
                        bbox = form_dict[tname]
                    else:
                        bbox = None

                    gt.append((img_path, transcr, bbox))

            return gt

        info = gather_iam_info(self, segmentation_level)

        previous_img_path = None

        data = []
        for i, (img_path, transcr, bbox) in enumerate(info):
            transcr = transcr.replace("|", " ")

            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            try:
                if segmentation_level == 'fword':
                    if img_path != previous_img_path:
                        timg = img_io.imread(img_path + '.png')
                        timg = 1 - timg.astype(np.float32) / 255.0
                        previous_img_path = img_path

                    # enlagre bbox
                    xs, ys, w, h = bbox[:]
                    wpad = int(2.5 * w / len(transcr))

                    nxs = max(0, xs - wpad)
                    nxe = min(timg.shape[1], xs + w + wpad)

                    hpad = 16
                    nys = max(0, ys - hpad)
                    nye = min(timg.shape[0], ys + h + hpad)

                    img = timg[nys:nye, nxs: nxe]
                    img = image_resize(img, height=img.shape[0] / 2)

                    bbox = [(ys - nys) // 2, (xs - nxs) // 2, h // 2, w // 2]
                else:
                    timg = img_io.imread(img_path + '.png')
                    img = 1 - timg.astype(np.float32) / 255.0
                    img = image_resize(img, height=img.shape[0] // 2)

                    if segmentation_level == 'form':
                        bbox = [(np.asarray(tt[0]) // 2, tt[1]) for tt in bbox]

                    if bbox is None:
                        bbox = [0, 0, img.shape[0], img.shape[1]]
            except:
                continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue
            #data += [(img, transcr.replace("|", " "))]
            data += [(img, transcr, bbox)]

        return data