from fastai.vision.all import *
import gradio as gr

def is_zespri(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('not zespri', 'zespri')

def classify_image(inp):
    pred,pred_idx,probs = learn.predict(inp)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['zespri.jpg', 'not_zespri.jpg']

iface = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
iface.launch()