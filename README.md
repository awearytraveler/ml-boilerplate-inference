# Generic Boiler Plate Code with Tensorflow

## Training

This code contains a simple implementation of a classifier.
In order to train a model with a ResNet-50 base use the train.py script as follows:

```
python train.py -f <path to your dataset>
```

## Inference

Inference can be done by either using the inference script. In order to run an evaluation on the test set, the code expects the image to be in a similar format compared to the training data. 

```
[Inference]

python inference.py -p <path to image>

[Evaluation]

python inference.py -p <path to dir> --evaluate

```

or a server can be launched and inference can be obtained from a post request

```
uvicorn main:app --reload
```

Example post request

```
[CURL]

curl -X POST -H 'Accept: */*' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 102' -H 'Content-Type: application/json' -d '{"url": "https://kristineskitchenblog.com/wp-content/uploads/2021/04/apple-pie-1200-square-592-2.jpg"}' http://127.0.0.1:8000/image/


[Python]
import requests
res = requests.post('http://127.0.0.1:8000/image/', json={'url':'https://kristineskitchenblog.com/wp-content/uploads/2021/04/apple-pie-1200-square-592-2.jpg'})
print(f"Output: {res.json()['message']})
```
