import os, sys
import cv2, numpy as np

from skimage.measure import compare_ssim
import cv2
import datetime
import math
from pyzbar.pyzbar import decode
import pyzbar.pyzbar as pyzbar

from myImageSearch.hashing import hamming
from imutils import paths
import argparse
import imagehash
import pickle
import vptree
import PIL
from PIL import Image
import cv2, sys
import copy
from sr_model import resolve_single
from sr_model.edsr import edsr
from sr_utils import load_image, plot_sample
from sr_model.wdsr import wdsr_b
from PIL import Image
import matplotlib
from scipy import ndimage, misc
import shutil
from distutils.dir_util import copy_tree

from project_config import temp_db_path
print('project config:', temp_db_path)
# from utils import detector_utils as detector_utils
# import tensorflow as tf

# detection_graph, sess = detector_utils.load_inference_graph()

wand = None

def resizePercent( img, percent):
    # print('Original Dimensions : ', img.shape)
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
            # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def getSubFolderList(path):
    return [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]

def unsharp_mask(image, kernel_size=(3,3), sigma=0.1, amount=0.5, threshold=0):
    import cv2 as cv
    # cv2.imshow('Orginal',image)
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    #cv2.imshow('unsharpMask',sharpened)
    #cv2.waitKey(0)
    return sharpened

def detectHand1(image_np):
    detected = False
    num_hands_detect = 2
    score_thresh = 0.15
    # image = cv2.imread('/home/ecologix/Desktop/hand.jpg')
    # img_w,img_h,img_c = image.shape
    # image_np = resizePercent(image_np, 30)
    img_w, img_h, img_c = image_np.shape
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
        return detected
    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    #  Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
    # start_time = datetime.datetime.now()
    boxes, scores = detector_utils.detect_objects(image_np,
                                                  detection_graph, sess)
    if any(s > score_thresh for s in scores):
        print('Hand Dettected..')
        detected = True
        return detected
    else:
        return detected

    # # # draw bounding boxes on frame
    # detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
    #                                  scores, boxes, img_h, img_w,
    #                                  image_np)
    # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    # print('time taken for 1 image: ',elapsed_time)
    # cv2.imshow('img',cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

def inRangeOfZero(number1, number2):
    if abs(number1-number2) <= 70000:
        return True
    else:
        return False
def changeDiffChack(number1, number2):
    if abs(number1-number2) >= 10000:
        return True
        # Product may Exist
    else:
        return False
#def clahe(img, clip_limit=3.0, grid_size=(8,8)):
def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    im = clahe.apply(img)
    #cv2.imshow('hh, im clahe', im)
    return im
# b, g, r = cv2.split(result)
# g = clahe(g, 5, (3, 3))
def getHighContrast(img):
    # img = cv2.imread('/home/ecologix/Desktop/temp-DB/product-0/testimg1.jpg', 1)
    # cv2.imshow("img",img)

    #-----Converting image to LAB Color model
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    return final

def point_inside_polygon(sx1,sy1,ex2,ey2,  px, py):
    f1 = False
    f2 = False
    f3 = False
    f4 = False
    if px >= sx1:
        f1 = True
    # else:
    #     print(f'px{px}!>=r{sx1}')
    if px <= ex2:
        f2 = True
    # else:
    #     print(f'px{px}!<=r{ex2}')
    if py >= sy1:
        f3 = True
    # else:
    #     print(f'py{py}!>=r{sy1}')
    if py <= ey2:
        f4 = True
    # else:
    #     print(f'py{py}!<=r{ey2}')
    # print(f'Points in poly f1:{f1}\nf2:{f2}\nf3:{f3}\nf4:{f4}')
    # print('res:', f1 and f2 and f3 and f4)
    return px >= sx1 and px <= ex2 and py >= sy1 and py <= ey2
def check_all_points_in_image(x,y,h,w, imgHeight, imgWidth):
    all_inside = False
    st_point_in_poly = point_inside_polygon(0,0,imgHeight,imgWidth, y, x) #st_point=  [x,y]
    # print('start in poly',st_point_in_poly,0,0,imgHeight,imgWidth, y, x)
    # print('')
    end_point_in_poly = point_inside_polygon(0,0,imgHeight,imgWidth, w+y, x+h) #end_point= [x+h,w+y]
    # print('end in poly',end_point_in_poly,0,0,imgHeight,imgWidth, w+y,x+h)
    if st_point_in_poly and end_point_in_poly:
        return True
    else:
        return False
def superRes(img):
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights('sr_weights/edsr-16-x4/weights.h5')

    # model = wdsr_b(scale=4, num_res_blocks=32)
    # model.load_weights('weights/wdsr-b-32-x4/weights.h5')

    # lr = cv2.imread('/home/line/Desktop/bq.jpg')
    lr = img
    sr = resolve_single(model, lr)

    # print('TypeOFSR_image:', type(sr))
    # print('ShapeOfSR:', sr.shape)
    # matplotlib.image.imsave('name.jpg', sr.numpy())
    return sr.numpy()

this_barcodeFlag = False
def rotate_enhance_getBarcodes(imgs):
    global this_barcodeFlag
    # print('In image rotate function')
    barcodes = []
    for img in imgs:
        # refined_img = cv2.bilateralFilter(img, 75, 50, 30)
        # refined_img = unsharp_mask(img, sigma=1, amount=0.5)
        # kernel = np.array([[0, -1, 0],
        #                    [-1, 5, -1],
        #                    [0, -1, 0]])
        #
        # # Sharpen image
        # refined_img = cv2.filter2D(img, -1, kernel)
        # refined_img = img
        refined_img = superRes(img)
        # gray = cv2.cvtColor(refined_img, cv2.COLOR_BGR2GRAY)
        # ret3, refined_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        angles = [90,180,30,45,60,270]
        for a in angles:
            rotated = ndimage.rotate(refined_img, a, reshape=True)  # just rotate
            print('bq-image rotated shape:',rotated.shape)
            red_bqs = pyzbar.decode(rotated)
            if len(red_bqs) < 1:
                print('NOT detected BQs', len(red_bqs), ' at angle:', a)
                # cv2.imwrite('/home/ecologix/Desktop/bq.jpg',rotated)
                # cv2.imshow('org', img)
                # cv2.imshow('No bq',rotated)
                # cv2.waitKey(0)
                continue
            # loop over the detected barcodes
            print('detected BQs',len(red_bqs),' at angle:',a)
            # cv2.imshow('enhanced', rotated)
            # cv2.waitKey(0)
            # print('in rotate.fun _total:',len(red_bqs))
            for bq in red_bqs:
                # (x, y, w, h) = barcode.rect
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                barcodeData = bq.data.decode("utf-8")
                barcodeType = bq.type
                # print('barcode Type:',barcodeType)
                if (barcodeData, barcodeType) not in barcodes and barcodeType == 'CODE128':
                    # totBarcodes = totBarcodes + 1
                    barcodes.append((barcodeData, barcodeType))
                    this_barcodeFlag = True
                    return barcodes
    print('rotate.fn returning BQs',barcodes)
    if len(barcodes) >= 1:
        return barcodes
    else:
        return []

arg_config = '/home/line/Desktop/bilal/experiments/longVideoProcessing/yolov3-custom7000.cfg'
arg_weights = '/home/line/Desktop/bilal/experiments/longVideoProcessing/yolov3-custom7000_6000.weights'
arg_names = '/home/line/Desktop/bilal/experiments/longVideoProcessing/custom.names'
# arg_image = '/home/line/Desktop/TEST_images/testYOLO13.jpg'
# Load the network
net = cv2.dnn.readNetFromDarknet(arg_config, arg_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def detectHandBq(img):
    global arg_names, arg_config,arg_weights, net,layers,output_layers
    CONF_THRESH, NMS_THRESH = 0.3, 0.5


    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    # img = cv2.imread(arg_image)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    # indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
    try:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
        print('Tot Detections:',len(indices))
    except:
        print('No Hand/BQ found')
        return None

    # Draw the filtered bounding boxes with their class to the image
    # with open(arg_names, "r") as f:
        # classes = [line.strip() for line in f.readlines()]
    classes = ['hand', 'barcode']
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))

    final_classes = []
    final_boxes = []
    final_confidences = []
    # print('confidences all:',confidences)

    for index in indices:
        x, y, w, h = b_boxes[index]
        final_boxes.append([x, y, x+w, y+h])
        final_classes.append(classes[class_ids[index]])
        print('labels:',classes[class_ids[index]])
        # print('classID:',classes[class_ids[index]])
        # print('Conf:',confidences[index])
        final_confidences.append(confidences[index])
        # print('P in ploy',point_inside_polygon(0,0,height, width,x,y))
        # img = cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
        # img = cv2.circle(img, (x + w, y + h), radius=4, color=(0, 0, 255), thickness=-1)
        # if check_all_points_in_image(x, y, w, h, height, width):
        #     pass
        #     # print('All Points in poly')
        #     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #     # cv2.putText(img, classes[class_ids[index]], (x + 5, y + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        # else:
        #     print('Some points not in poly')

    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    final_result = zip(final_classes,final_boxes,final_confidences)
    return final_result

def processVideo(videopath):
    global this_barcodeFlag
    frame_count = 0
    product_exists = False
    n_white_pix_zero = 0
    products_count = 0
    temp_db_path = '/home/line/Desktop/temp-DB/'
    kernel = np.ones((7,7),np.uint8)
    kernele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kerneld = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernelc = np.ones((5, 5), np.uint8)

    zero_frame = None               # /home/ecologix/Desktop/AppDemp.MOV    /home/ecologix/Desktop/vid.mp4
    video_capture = cv2.VideoCapture(videopath)

    ## setting first frame
    ret, zero_frame = video_capture.read()
    zero_frame = resizePercent(zero_frame,50)
    for i in range(30):
        ret, this_frame = video_capture.read()
        this_frame = resizePercent(this_frame, 50)
        if this_frame is None :
            print('First frame NOT set')
            sys.exit()
        n_white_pix = np.sum(this_frame <= 150)
        # print('n_white_pix_in_zero Frame:', n_white_pix_zero)
        if n_white_pix < n_white_pix_zero:
            zero_frame = this_frame
            n_white_pix_zero = np.sum(zero_frame <= 150)
            print('n_white_pix_in_zero Frame:',n_white_pix_zero)
    zero_frame = cv2.rotate(zero_frame, cv2.ROTATE_180)
    # print(zero_frame.shape)     #   (1080, 1920, 3)
    # zero_frame = getHighContrast(zero_frame)
    grayZ = cv2.cvtColor(zero_frame, cv2.COLOR_BGR2GRAY)
    grayL = grayZ
    # cv2.imshow('zero frame',zero_frame)
    # cv2.waitKey(0)
    blur_score_at_zero = math.ceil(variance_of_laplacian(zero_frame))
    print('blur scrore at zero frame:', blur_score_at_zero)

    while (True):
        barcodeFlag = False
        barcodeData = None

        ret, current_frame = video_capture.read()
        #current_frame= src
        if current_frame is None :
            print('null frame')
            break
        # print(frame_count,' frame: ')

        frame_count = frame_count+1
        if frame_count % 20 != 0:
            # print('frame skipped')
            continue

        # if frame_count <= 3520:
        #     continue
        print('1.Outer -----------',frame_count)

        current_frame = resizePercent(current_frame, 50)
        current_frame = cv2.rotate(current_frame, cv2.ROTATE_180)
        # cv2.imshow("1 current frame ", current_frame)
        # cv2.waitKey(0)

        blur_score = variance_of_laplacian(current_frame)
        print('1.[INFO] Blur score: ',blur_score)
        # if blur_score < 100:     # less_score = more blury image
        #     print('-[INFO] Blury Image')


        # if detectHand1(current_frame):
        #     print('1. [INFO] Hand Detected.. Dropping Frame')
        #     continue
        ###################################################
        #####################
        # un-COMMENT AAAAAAAAAAAAAAAAAAAAAAAAAAAABOVE LINE//
        # TESTING YYYYYYYYYOLO hand
        result = detectHandBq(current_frame)
        if result is not None:
            handFlag = any(cl == 'hand' for cl, rec, cf in copy.deepcopy(result))
            print('handFlag:', handFlag)
            if handFlag == True:
                # now = datetime.datetime.now()
                # time = str(now.strftime("%H:%M:%S"))
                # cv2.imwrite('/home/line/Desktop/allHands/'+time+'.jpg',current_frame)
                continue
        ###################################################
        ###################################################
        ###################################################
        ###################################################

        grayC = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # diff = cv2.absdiff(grayZ,grayC)
        # print('diff shape: ',diff.shape)
        # sys.exit()
        # HC_diff = clahe(diff,2,(8,8))
        # diff = (diff +50).astype("uint8")
        # UserWarning: DEPRECATED: skimage.measure.compare_ssim has been moved to skimage.metrics.structural_similarity.
        # It will be removed from skimage.measure in version 0.18.
        (ssim_score, ssim_diff) = compare_ssim(grayZ, grayC, full=True)
        ssim_diff = (ssim_diff * 255).astype("uint8")
        print("1. SSIM: {}".format(ssim_score))

        # cv2.imshow('diff', diff)
        # cv2.imshow('ssim_diff',ssim_diff)
        # cv2.imshow('HC_diff', HC_diff)
        reversed_ssim = cv2.bitwise_not(ssim_diff)
        # cv2.imshow('reversed_ssim',reversed_ssim)

        ret_, thresh0 = cv2.threshold(reversed_ssim, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('thresh0 ', thresh0)
        inverted_change = np.sum(thresh0 >= 150)
        print('1. inverted change: ', inverted_change)
        # if inverted_change < 121000:
        #                             # smaller sized image : 1120
        #                             # Orignal size image :  12100
        #     print('NOT- frame {} skipped (movement) '.format(frame_count), 'change: ', inverted_change)
        if ssim_score >= 0.975:
            print('1. No Product found')
            continue
        else:
            print('1. product Probable')
        print('1. change: ',abs((inverted_change-n_white_pix_zero)))
        if changeDiffChack(inverted_change,n_white_pix_zero):
            print('======== Product found:1 ===========')
            product_exists = True

        roi = None

        while (product_exists):
            prod_folder_name = None
            barcodes = []
            ret, current_frame = video_capture.read()
            # current_frame= src
            if current_frame is None:
                print('null frame')
                break
            # print(frame_count,' frame: ')
            frame_count = frame_count + 1
            # detectHand1(current_frame)
            if frame_count % 25 != 0:
                # print('frame skipped')
                continue
            print('------------ 2.Inner -----------', frame_count)
            org_current_frame = current_frame
            org_current_frame = cv2.rotate(org_current_frame, cv2.ROTATE_180)

            current_frame = resizePercent(current_frame, 50)
            current_frame = cv2.rotate(current_frame, cv2.ROTATE_180)
            grayC = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            _sigma = 0.3
            _amount = 0.3
            totBarcodes = 0

            (ssim_score, ssim_diff) = compare_ssim(grayZ, grayC, full=True)
            ssim_diff = (ssim_diff * 255).astype("uint8")
            print("SSIM: {}".format(ssim_score))

            # cv2.imshow('2 current', current_frame)

            # cv2.imshow('ssim_diff',ssim_diff)
            # cv2.imshow('HC_diff', HC_diff)
            reversed_ssim = cv2.bitwise_not(ssim_diff)
            # cv2.imshow('reversed_ssim', reversed_ssim)

            ret_, thresh0 = cv2.threshold(reversed_ssim, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow('thresh0 ', thresh0)
            inverted_change = np.sum(thresh0 >= 150)
            print('2. inverted change: ', inverted_change)
            # if inverted_change < 12100:
            #                         # smaller sized image : 1120
            #                         # Orignal size image :  12100
            #     print('NOT- frame {} skipped (movement) '.format(frame_count), 'change: ', inverted_change)
            if ssim_score >= 0.975:
                print('Product Removed..')
                print('\n\n\n\n\n\n\n\n\n\n BROKE LOOP \n\n\n\n\n\n\n\n\n\n\n\n\n\n')
                product_exists = False
                products_count += 1
                barcodeFlag = False
                this_barcodeFlag = False
                barcodes = []
                # sys.exit()

            blur_score = int(variance_of_laplacian(current_frame))
            blur_score = blur_score + blur_score_at_zero
            print('2. [INFO] Blur score: ', blur_score)
            # if blur_score < 380:  # less_score = more blury image
            #     print('-[INFO]2 Blury Image')
                # continue

            ###################################################
            #####################
            # UNCOMMENT AAAAAAAAAAAAAAAAAAAAAAAAAAAABOVE LINE//
            # TESTING YYYYYYYYYOLO hand

            result = detectHandBq(current_frame)
            if result is not None:
                handFlag = any(cl == 'hand' for cl, rec, cf in copy.deepcopy(result))
                bqFlag = any(cl == 'barcode' for cl, rec, cf in copy.deepcopy(result))
                # if bqFlag==True and barcode_redFlag == False:
                #     print('barcode Detected..',bqFlag)
                #     print('Result: ',list(result))
                #     # sys.exit()
                print('handFlag:', handFlag)
                # if handFlag == True and bqFlag == False:
                if handFlag == True:
                    # now = datetime.datetime.now()
                    # time = str(now.strftime("%H:%M:%S"))
                    # cv2.imwrite('/home/ecologix/Desktop/allHands/' + time + '.jpg', current_frame)
                    continue
                # if handFlag == True and bqFlag == True and this_barcodeFlag == True:
                #     continue
                # if bqFlag == True and this_barcodeFlag == False:
                #     pass
                barcode_img_recs = [rec for cl, rec, cf in result if cl == 'barcode' and cf >= 0.3]
                print('barcode regions:', barcode_img_recs)
                barcode_imgs = []
                for bq in barcode_img_recs:
                    x, y, w, h = bq
                    # print('rec:',x, y, w, h)
                    bq_img = current_frame[y:h, x:w]
                    barcode_imgs.append(bq_img)
                    # cv2.imshow('bq', bq_img)
                    # cv2.waitKey(0)
                if this_barcodeFlag == False:
                    barcodes = rotate_enhance_getBarcodes(barcode_imgs)
                print('Red barcodes:', barcodes)
            ###################################################
            ###################################################
            ###################################################
            ###################################################

            if len(barcodes) >= 1:
                barcode = barcodes[0]
                barcodeData = barcode[0]
                print('barcodeData Received..',barcodeData)
                prod_folder_name = 'barcode-'+str(barcodeData)
                barcode_redFlag = True
                print('barCODE FOUNDD>>>>>> ',barcode_redFlag)
                if barcodeData is not None and product_exists == True:
                    # print('BARCODE FOUND !! ',barcode_redFlag)
                    newBarcodeFolder = temp_db_path +'product'+str(products_count)+'-'+ str(barcodeData)
                    print('BQ Dir made:::',newBarcodeFolder)
                    if not os.path.exists(temp_db_path +'product'+str(products_count)+'-'+ str(barcodeData)):
                        os.makedirs(temp_db_path +'product'+str(products_count)+'-'+ str(barcodeData))
                # sys.exit()
                # if prod_folder_name is not None:
                #     if not os.path.exists(temp_db_path + prod_folder_name ):
                #         os.makedirs(temp_db_path + prod_folder_name)

            if barcodeData == None or len(barcodes)<1 and product_exists == True:
                barcode_redFlag = False
                # barcodeData = 'product'+str(products_count)
                if not os.path.exists(temp_db_path + 'product' + str(products_count)):
                    os.makedirs(temp_db_path +'product'+ str(products_count))
                    print('Made new Prod Dir',temp_db_path +'product'+ str(products_count))
                    # sys.exit()

            # if not os.path.exists(temp_db_path + 'product' + str(products_count)):
            #     os.makedirs(temp_db_path + 'product' + str(products_count))

            # product_exists = True
            # median_blur = cv2.medianBlur(diff, 3)
            # cv2.imshow('median_blur', median_blur)
            thresh0 = cv2.adaptiveThreshold(thresh0, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
            # cv2.imshow('thresh1 ', thresh1)
            eroded = cv2.erode(thresh0, kernele, iterations=2)
            closing = cv2.morphologyEx(thresh0, cv2.MORPH_CLOSE, kernelc, iterations=2)
            # cv2.imshow('closing 1 ', closing)
            _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # _,contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 3500:
                continue
            rect = cv2.boundingRect(c)
            # print('Rect: ',rect)
            x, y, w, h = rect
            cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            roi = np.zeros([y+h, x+w, 3], dtype=np.uint8)
            roi = current_frame[y:y + h, x:x + w]
            # cv2.drawContours(current_frame, [rect], -1, (0,0, 255),7)
            print('Contour Area= ',cv2.contourArea(c))
            blur_score_in_roi = int(variance_of_laplacian(roi))
            print('Blur in ROI= ', blur_score_in_roi)

            date_time_stamp = str(datetime.datetime.now())
            raw_date_time_list = date_time_stamp.split(' ')
            date_time_stamp = raw_date_time_list[1]
            date_time_stamp = date_time_stamp.replace('.', ':')
            # date_time_stamp = date_time_stamp[:date_time_stamp.rfind('.')]
            # print('dateTime Now:', date_time_stamp)
            # sys.exit()

            av_color = avgColor(roi)
            # if blur_score_in_roi < 500:
            #     if av_color == 'b':
            #         continue
            if barcode_redFlag == True and roi is not None and blur_score_in_roi > 340:
                image_write_path = temp_db_path + 'product' + str(products_count) + '-' + str(barcodeData)
                print('writing BQ image to\n:',image_write_path + '/'+ str(blur_score_in_roi)+'-'+av_color+'-' + 'barcode' + barcodeData + '.jpg')
                cv2.imwrite(image_write_path+'/'+ str(blur_score_in_roi) +'-'+av_color+ '-' + 'barcode' + barcodeData + '.jpg', roi)
                continue

            if roi is not None and blur_score_in_roi > 340:
                image_write_path = temp_db_path+'product'+str(products_count)
                # if not os.path.exists(image_write_path):
                #     print('Dir made on image write..')
                #     os.makedirs(image_write_path)
                print('Image written path:\n', image_write_path+'/'+str(blur_score_in_roi)+'-'+av_color+'-'+date_time_stamp+'.jpg')
                # temp_db_path + 'product' + str(products_count) + '/' + str(blur_score_in_roi) + '-' +date_time_stamp + '.jpg'
                cv2.imwrite(image_write_path+'/'+str(blur_score_in_roi)+'-'+av_color+'-'+date_time_stamp+'.jpg', roi)
            # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
# processVideo()

def getParent(path):        # returns folder of from path
    return str(os.path.split(os.path.split(path)[0])[1])
def getLastFolder(path):    # returns last element from path
    import os
    return os.path.basename(os.path.normpath(path))
def get_grays(image, width, height):

    if isinstance(image, (tuple, list)):
        if len(image) != width * height:
            raise ValueError('image sequence length ({}) not equal to width*height ({})'.format(
                len(image), width * height))
        return image

    if PIL is not None and isinstance(image, PIL.Image.Image):
        gray_image = image.convert('L')
        small_image = gray_image.resize((width, height), PIL.Image.ANTIALIAS)
        # print('pil ok')
        return list(small_image.getdata())

    # else:
    #     raise ValueError('image must be a wand.image.Image or PIL.Image instance')
def dhash_row_col(image, size=8):
    width = size + 1
    grays = get_grays(image, width, width)

    row_hash = 0
    col_hash = 0
    for y in range(size):
        for x in range(size):
            offset = y * width + x
            row_bit = grays[offset] < grays[offset + 1]
            row_hash = row_hash << 1 | row_bit

            col_bit = grays[offset] < grays[offset + width]
            col_hash = col_hash << 1 | col_bit

    return (row_hash, col_hash)
def dhash_int(image, size=8):
    row_hash, col_hash = dhash_row_col(image, size=size)
    return row_hash << (size * size) | col_hash
def load_image(filename):
    if wand is not None:
        return wand.image.Image(filename=filename)
    elif PIL is not None:
        return PIL.Image.open(filename)
    else:
        sys.stderr.write('You must have wand or Pillow/PIL installed to use the dhash command\n')
        sys.exit(1)
def trainHashes(path):
    print('making hash for root:',path)
    size_ = 8
    master_root = path
    tree_path = master_root+'vptree.pickle'
    hash_path = master_root+'hashes.pickle'
    # grab the paths to the input images and initialize the dictionary
    # of hashes
    imagePaths = list(paths.list_images(master_root))
    hashes = {}

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        # image = cv2.imread(imagePath)
        image1 = load_image(imagePath)
        h = dhash_int(image1, size=size_)
        # print('hash: ', h)
        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(imagePath)
        hashes[h] = l

    # build the VP-Tree
    print("[INFO] building VP-Tree...")
    points = list(hashes.keys())
    tree = vptree.VPTree(points, hamming)

    # uilding an Image Hashing Search Engine with VP-Trees and OpenCVPython
    # serialize the VP-Tree to disk
    print("[INFO] serializing VP-Tree...")
    f = open(tree_path, "wb")
    f.write(pickle.dumps(tree))
    f.close()

    # serialize the hashes to dictionary
    print("[INFO] serializing hashes...")
    f = open(hash_path, "wb")
    f.write(pickle.dumps(hashes))
    f.close()

def avgColor(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    Y = 0.2126 * avg_color[0] + 0.7152 * avg_color[1] + 0.0722 * avg_color[2]
    # Then check if the value is nearer to 0 or to 255 and choose black or white accordingly
    if Y >= 128:
        # print('white')
        return 'w'
    else:
        # print('black')
        return 'b'
    # print('avg color:', avg_color)
    # return avg_color
def removeDuplicates(root):
    # root = '/home/ecologix/Desktop/temp-DB/testfolder'
    #
    # root = root + '/'
    # print('root:', root)
    trainHashes(root)
    tree_path = root+'vptree.pickle'
    hash_path = root+'hashes.pickle'
    tree = pickle.loads(open(tree_path, "rb").read())
    hashes = pickle.loads(open(hash_path, "rb").read())
    # load the VP-Tree and hashes dictionary
    print("[INFO] loading VP-Tree and hashes...")
    distance_threshold = 17
    size_ = 8
    imagePaths = list(paths.list_images(root))
    for imagePath in imagePaths:
        # load the input image
        # print("[INFO] processing image {}/{}".format(i + 1,
        #                                              len(imagePaths)))
        image = cv2.imread(imagePath)
        if image is None:
            print('NONE image')
            continue
        print('imagePath:',imagePath)
        # cv2.imshow('Query', image)

        pil_img = Image.fromarray(image.astype('uint8'), 'RGB')

        queryHash = dhash_int(pil_img, size=size_)
        # queryHash = dhash_int(PIL.Image.open(queryImg), size=size_)
        print("[INFO] performing search...", queryHash)
        results = tree.get_all_in_range(queryHash, distance_threshold)
        if len(results) <= 0:
            print('No match')
            # sys.exit()
        results = sorted(results)

        mapper = []
        # sys.exit()
        print('Input Image: ',imagePath)
        print('Matches: ',len(results))
        to_remove = []
        for (d, h) in results:
            # grab all image paths in our dataset with the same hash
            path = hashes.get(h, [])

            if imagePath in path:
                path.remove(imagePath)
                # print('___saved current image to removal___')
            if len(path) <= 0:
                continue
            # print('Result Path:', path)

            to_remove.append([path[x] for x in range(len(path)) if len(path[x])>0])
                # continue
            # print(path[0] for x in range(len(path)))
            # to_remove.append([path[x] for x in range(len(path))] )
        to_remove = [item for sublist in to_remove for item in sublist]
        print('hashing-images To Remove: ',to_remove)
        for removal_image in to_remove:
            if removal_image in imagePaths:
                imagePaths.remove(removal_image)
            if os.path.exists(removal_image):
                os.remove(removal_image)
            imagePaths = list(paths.list_images(root))
            print('removed:',removal_image)
    os.remove(tree_path)
    os.remove(hash_path)

def dir_size(path):
    size = 0
    for x in os.listdir(path):
        if not os.path.isdir(os.path.join(path,x)):
            size += os.stat(os.path.join(path,x)).st_size
        else:
            size += dir_size(os.path.join(path,x))

    return size/1024                #| returns in Kbs
def mergeFolders(folders_path):
    subFolders = getSubFolderList(folders_path)
    print('subfolders:',subFolders)

    prodNameFolders = [x for x in subFolders if '-' not in x]
    print('ProductName Folders:',prodNameFolders)
    barcodeFolders = [x for x in subFolders if '-' in x]
    print('BarcodeName Folders:', barcodeFolders)

    for barcodeFolder in barcodeFolders:
        for prodNameFolder in prodNameFolders:
            if prodNameFolder in barcodeFolder:
                print('move Folder:',prodNameFolder,'to ->',barcodeFolder)
                splitStr = barcodeFolder.split('-')
                barcodeName = splitStr[1]
                src1Dir = folders_path + '/' + prodNameFolder
                src2Dir = folders_path + '/' + barcodeFolder
                dstDir = folders_path+ '/' + str(barcodeName)

                if not os.path.exists(dstDir):
                    print('made Dir:', dstDir)
                    os.makedirs(dstDir)

                copy_tree(src1Dir, dstDir)
                print('moved ', src1Dir, 'to ->', dstDir)
                copy_tree(src2Dir, dstDir)
                print('moved ', src2Dir, 'to ->', dstDir)
                shutil.rmtree(src1Dir)
                print('Deleted dir', src1Dir)
                shutil.rmtree(src2Dir)
                print('Deleted dir', src2Dir)

def cleanDB():
    for root, _dirs, _files in os.walk(basePath):
        if root == basePath:
            continue
        subFolders = getSubFolderList(basePath)
        lastFolder = getLastFolder(root)
        if lastFolder not in subFolders:
            continue
        print('root:',root)
        mergeFolders(basePath)
        if os.path.exists(root):
            removeDuplicates(root)
        else:
            continue
        # removeBlured(root)
# cleanDB()

def main():
    videopath = '/home/line/Desktop/AppDemp.MOV'
    basePath = '/home/line/Desktop/temp-DB'
    processVideo(videopath)
    # mergeFolders(basePath)
    # for root, _dirs, _files in os.walk(basePath):
    #     print('\n\n\nroot\n========\n:', root)
    #     if root == basePath:
    #         continue
    #
    #     folderSize = dir_size(root)
    #     if folderSize <= 10:  # size in Kbs
    #         shutil.rmtree(root)
    #         print('Removing empty folder:',root)
    #
    #     subFolders = getSubFolderList(basePath)
    #     lastFolder = getLastFolder(root)
    #     if lastFolder not in subFolders:
    #         continue
    #
    #     if os.path.exists(root):
    #         removeDuplicates(root)
    #     else:
    #         continue
main()
