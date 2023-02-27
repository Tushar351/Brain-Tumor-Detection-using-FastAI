import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from fastai.vision.all import *
from fastai.vision.widgets import *

fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

dls = fields.dataloaders('../input/brain-tumor-classification-mri') # enter the dataset path

dls.vocab

dls.show_batch()

learn = vision_learner(dls,resnet152,metrics = [accuracy,error_rate])  # Download resnet152

learn.fine_tune(1)

learn.lr_find()

learn.fit_one_cycle(4,6e-5)

learn.recorder.plot_loss()

learn.show_results()

learn.unfreeze()

learn.fit_one_cycle(4,lr_max=slice(1e-5,6e-5))

learn.show_results()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(9, figsize=(15,10))

learn.export()

path = Path()
learn_inf = load_learner(path/'export.pkl')

learn.predict(path/'../input/brain-tumor-classification-mri/Testing/glioma_tumor/image(1).jpg')

btn_upload = widgets.FileUpload()
btn_run = widgets.Button(description='Classify')
out_pl = widgets.Output()
lbl_pred = widgets.Label()

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    
btn_run.on_click(on_click_classify)

VBox([widgets.Label('Select your MRI'),btn_upload, btn_run, out_pl, lbl_pred])
