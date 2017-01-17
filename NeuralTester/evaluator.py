from optparse import OptionParser
from texttable import Texttable
import os
import copy
import sys
import operator
import classifier
import cv2

def parseFilename(filename):
    return filename.split("_")[0]

def classify(model, imagePath):
    image = classifier.processImage(imagePath)
    image = classifier.image_to_feature_vector(image)
    return classifier.classify(model, image)

def generateConfusionTable(classes):
    return {key: [0] * len(classes) for key in classes}

def drawConfusionTable(classes, table):
    t = Texttable()
    results = [' ']
    results += classes
    results = [results]
    t.set_cols_width([10] * (len(classes)+1))
    for key, value in table.iteritems():
        results.append([key] + value)
    t.add_rows(results)
    sys.stdout.write(t.draw())
    sys.stdout.flush()

def confusionToCSV(classes, table):
    results = ['-']
    results += classes
    results = ";".join(results) + "\n"
    for key, value in table.iteritems():
        results += key + ";" + ";".join([str(x) for x in value]) + "\n"
    return results

def drawProgressBar(percent, errorRate, successRate, barLen = 50):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[%s] %.2f%%, errors => %.2f%%, success => %.2f%%" % (progress, percent * 100, errorRate * 100, successRate * 100))
    sys.stdout.flush()

if __name__ == '__main__':
    parser = OptionParser(usage="usage: evaluator --images /path/to/images --model model.tfl",
                          version="%prog 1.0")
    parser.add_option("-i", "--images",
                      dest="image_directory",
                      default="",
                      type="string",
                      help="Path to images to classify")
    parser.add_option("-m", "--modele",
                      dest="model_file",
                      type="string",
                      default="",
                      help="The TensorFlow model to evaluate")
    (options, args) = parser.parse_args()

    image_directory = options.image_directory
    model_file = options.model_file

    print "Loading TensorFlow Model (" + str(model_file) + ")..."
    model = classifier.createModel(model_file)
    bad_files = []
    attempts = 0
    success = 0
    errors = 0
    classes = ['bomb', 'danger', 'fire', 'car', 'forbidden', 'flash', 'dead', 'air', 'water', 'urgent', 'flag', 'warning', 'parking', 'man']
    results = generateConfusionTable(classes)
    for dirname, _, filenames in os.walk(image_directory):
        print "Analyzing " + str(len(filenames)) + " Images..."
        for filename in filenames:
            if ".png" not in filename: continue
            #drawProgressBar(float(attempts)/float(len(filenames)), float(errors)/float(attempts+1), float(success)/float(attempts+1))
            drawConfusionTable(classes, results)
            imagePath = os.path.join(dirname, filename)
            classe = parseFilename(filename)
            result = classify(model, imagePath)
            result = max(result.iteritems(), key=operator.itemgetter(1))[0]
            results[classe][classes.index(result)] += 1
            attempts += 1
            if result == classe:
                success += 1
            else:
                bad_files.append(filename)
                errors += 1
    #drawProgressBar(1, float(errors)/float(attempts), float(success)/float(attempts))
    drawConfusionTable(classes, results)
    print "Error rate: ", (float(errors)/float(attempts)) * 100, "%"
    print "Accuracy rate: ", (float(success)/float(attempts)) * 100, "%"
    print confusionToCSV(classes, results)
    print "Files that generated errors: "
    print bad_files
