def simple(classifier, test_data, test_labels, test_images, plots):

    correct = 0
    total = len(test_data)
    print("Number of tests: ", total)

    false_positive = None
    false_negative = None

    false_positives = 0
    false_negatives = 0 
    for index, image in enumerate(test_data):
        prediction = classifier.predict(image)
        label = test_labels[index]
        
        false_positives += (prediction == 1 and label == 0)
        false_negatives += (prediction == 0 and label == 1)
        
        if prediction == 1 and label == 0:
            false_positive = test_images[index]
        
        if prediction == 0 and label == 0:
            false_negative = test_images[index]
            
        correct += 1 if classifier.predict(image) == test_labels[index] else 0
        

    fig, (fp, fn) = plots

    fp.imshow(false_positive)
    fn.imshow(false_negative)

    print("Success rate: {}".format(correct / total))
    print("False positives {}".format(false_positives / total))
    print("False negatives {}".format(false_negatives / total))
