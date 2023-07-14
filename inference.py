import pathlib
import PIL
import tensorflow_hub as hub
import tensorflow as tf
import pickle
import argparse
import os

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

  model = tf.keras.models.load_model(export_dir)
  return model

def evaluateModel(data_path, model):

  if not os.path.isdir(data_path):
    raise Exception("Expected directory similar to train_data")
  
  test_ds = tf.keras.utils.image_dataset_from_directory(
          data_path,
          validation_split=0.999,
          subset="validation",
          seed=123,
          image_size=(224, 224),
          batch_size=1
        )

  out = model.evaluate(test_ds)

  return out

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description = "Inference Module")
  parser.add_argument("-p", "--path", help = "Input File Path")
  parser.add_argument("-e", "--evaluate", action=argparse.BooleanOptionalAction, help = "Evaluation mode")

  args = parser.parse_args()
  if args.path:

    model = loadModel('saved_model/my_model')
    class_names = pickle.loads(open('labels.pickle', "rb").read())

    if args.evaluate:
      out = evaluateModel(args.path, model)
      print(f"Evalution Results\n\nLoss: {out[0]}\nAccuracy: {out[1]}")

    else:
      out = runInference(args.filePath,model)
      result = class_names[out.argmax(axis=1)[0]]
      print(f"Inference Results: {result}")

