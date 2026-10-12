import urllib.request

trainSet = "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz"
trainLabels = "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz"

testSet = "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz"
testLabels = "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz"

def downloadFile(*urls: str):
    for url in urls:
        destination: str = url[url.rfind('/') + 1:]
        destination = f"data/{destination}"

        urllib.request.urlretrieve(url, destination)

downloadFile(trainSet, trainLabels, testSet, testLabels)

