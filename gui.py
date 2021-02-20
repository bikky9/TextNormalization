from tkinter import *
from parameters import parser, change_args
from lib.data.Tweet import Tweet
from lib.data.DataLoader import create_data
import lib
import numpy as np
from main import train_char_model


def normalize(noisy_text):
    var = noisy_text
    tweets = [Tweet(var.split(), var.split(), '1', '1') for i in
              range(2)]  # suboptimal but works with minimal changes
    test_data, test_vocab, _ = create_data(tweets, opt=opt, vocab=vocab, mappings=mappings)
    prediction = test_evaluator.eval(test_data)
    return prediction


def retrieve():
    noisy_text = input_text.get("1.0", "end-1c")
    normalized_text = normalize(noisy_text)
    output_text.delete("1.0", "end-1c")
    output_text.insert(END, normalized_text)


root = Tk()
root.geometry("640x360")
root.title("Text Normalization")

frame = Frame(root)
frame.pack()

label = Label(frame, text="Enter Noisy Text: ", font=("Arial", 25))
label.pack()
input_text = Text(frame, width=50, height=3, bg="light yellow")
input_text.pack(padx=5, pady=5)

Button = Button(frame, text="Submit", command=retrieve, )
Button.pack(padx=5, pady=5)
label = Label(frame, text="Normalized Text: ", font=("Arial", 25))
label.pack()
output_text = Text(frame, width=50, height=3, bg="light cyan")
output_text.pack(padx=5, pady=5)

# code for loading models using pre defined parameter froms params.npy

opt = np.load("params.npy", allow_pickle=True).item()
unk_model = train_char_model(opt) if (opt.input in ['hybrid', 'spelling']) else None
if (opt.input == 'spelling'): exit()
train_data, valid_data, test_data, vocab, mappings = lib.data.create_datasets(opt)
model, optim = lib.model.create_model((vocab['src'], vocab['tgt']), opt)
evaluator = lib.train.Evaluator(model, opt, unk_model)
test_evaluator = lib.train.Evaluator(model, opt, unk_model)
#

root.mainloop()
