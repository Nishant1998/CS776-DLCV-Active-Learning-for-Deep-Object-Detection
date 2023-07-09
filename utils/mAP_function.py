from yolo_utils import *

def getDataForMap(model_path):
    """ This function calculate mAP for test data given model path """
    test_images, test_labels = get_test()

    # get targets
    target = []
    preds = []

    # get data
    anchors = get_anchors()
    class_names = get_classes()
    num_anchors = len(anchors)
    num_classes = len(class_names)

    # load model
    yolo_model = load_trained_model(input_shape=(416, 416), num_classes=num_classes, load_pretrained=True,
                                    freeze_body=2, weights_path=model_path)
    # yolo_model = load_model(model_path, compile=False)
    iii = 0

    print("total test:", len(test_labels))
    for a in test_labels:
        iii = iii + 1

        #         if iii == 100:
        #             break
        print(iii)
        boxes = []
        labels = []
        for b in a:
            c = b.split(",")
            c = [int(i) for i in c]

            # bbox = [c[0],c[1],c[0]+c[2],c[1]+c[3]]

            boxes.append(c[:-1])
            labels.append(c[-1])

        # labels = [i-1 for i in labels]
        target.append(
            dict(
                boxes=torch.tensor(boxes),
                labels=torch.tensor(labels),
            )
        )

    iii = 0
    print(len(test_images))
    for names in test_images:
        iii = iii + 1

        #         if iii == 100:
        #             break
        print(iii)
        image = read_image(names)
        image_data = preprocess_image(image)  # (1, 416, 416, 3)

        yolo_outputs = yolo_model.predict(image_data)

        out_boxes, out_scores, out_classes, _ = yolo_eval(yolo_outputs, anchors, num_classes,
                                                          [image.size[1], image.size[0]], max_boxes=20,
                                                          score_threshold=.6, iou_threshold=.5)

        out_boxes = out_boxes.numpy()

        for i in range(len(out_boxes)):
            out_boxes[i][0] = max(0, np.floor(out_boxes[i][0] + 0.5).astype('int32'))
            out_boxes[i][1] = max(0, np.floor(out_boxes[i][1] + 0.5).astype('int32'))
            out_boxes[i][2] = min(image.size[1], np.floor(out_boxes[i][2] + 0.5).astype('int32'))
            out_boxes[i][3] = min(image.size[0], np.floor(out_boxes[i][3] + 0.5).astype('int32'))

        out_boxes = out_boxes[:, [1, 0, 3, 2]]

        preds.append(dict(
            boxes=torch.tensor(out_boxes),
            scores=torch.tensor(out_scores.numpy()),
            labels=torch.tensor(out_classes.numpy()),
        ))

    return preds, target