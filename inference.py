import pathlib
import PIL
import tensorflow_hub as hub
import tensorflow as tf
import pickle
import argparse

def runInference(input, model):
  
  data_init = tf.keras.Sequential([
    tf.keras.layers.Resizing(224,224),
  ])

  infer = model.signatures["serving_default"]
  if isinstance(input, str):
    input = PIL.Image.open(input)

  img = data_init(input)
  img = tf.expand_dims(img, 0)

  out = infer(img)['dense_1'].numpy()
  return out

def loadModel(export_dir):

  model = tf.saved_model.load(
        export_dir, tags=None, options=None
    )
  
  return model
  
if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description = "Inference Module")
  parser.add_argument("-f", "--filePath", help = "Input File Path")

  args = parser.parse_args()
  if args.filePath:
    model = loadModel('finetuned_model_export')
    class_names = pickle.loads(open('labels.pickle', "rb").read())

    out = runInference(args.filePath,model)
    result = class_names[out.argmax(axis=1)[0]]
    print(f"Inference Results: {result}")

