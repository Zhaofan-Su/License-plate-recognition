import cv2
import os
import sys
import codecs
import argparse
import numpy as np
import tensorflow as tf



parser = argparse.ArgumentParser(description="Car license detection and chars recognization")
parser.add_argument('-i', '--image', default='1.jpg', type=str, help='The image to test.')
parser.add_argument('--char_model', default='char-models/char-model-520.meta', type=str, help='The CNN net to recognize the chars.')
parser.add_argument('--license_model', default='license-models/license-model-520.meta', type=str, help='The CNN net to select the car license.')

args = parser.parse_args()


license_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '渝', '川', '鄂', '赣', '甘',
                '贵', '桂', '黑','沪','冀','吉', '津', '京', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '晋', '苏',
                '皖', '湘', '新', '豫', '粤', '云', '藏', '浙']


# def hist(img):
#     # 图像直方图均衡化
#     assert img.ndim == 2
#     hist = [0 for i in range(256)]
#     img_h = img.shape[0]
#     img_w = img.shape[1]

#     for row in range(img_h):
#         for col in range(img_w):
#             hist[img[row, col]]+=1
#     p =[hist[n]/(img_w * img_h) for n in range(256)]
#     p = np.cumsum(p)
#     for row in range(img_h):
#         for col in range(img_w):
#             img[row, col] = p[img[row, col]] * 255
#     return img


def verify_scale(rotate_rect):
    err = 0.4
    aspect = 4
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - err)
    max_aspect = aspect * (1 + err)
    theta = 30

    # 宽或高为0, 非矩形, 返回False
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False
    
    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1/r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if area > min_area and area < max_area and r > min_aspect and r < max_aspect:
        # 矩形的倾斜角度不超过theta
        if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or 
        (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
            return True
    return False


def img_transform(car_rect, image):
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2] == 0:
        return_flag = True
    if car_rect[2] == 90 and rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1] - rect_h / 2): int(car_rect[0][1] + rect_h / 2),
        int(car_rect[0][0] - rect_w / 2) : int(car_rect[0][0] + rect_w / 2)]
        return car_img

    car_rect =(car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    height_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point  = point
        if height_point[1] < point[1]:
            height_point = point
        if right_point[0] < point[0]:
            right_point = point
    
    # 正角度
    if left_point[1] <= right_point[1]:
        new_right_point = [right_point[0], height_point[1]]
        p1 = np.float32([left_point, height_point, right_point])
        p2 = np.float32([left_point, height_point, new_right_point])
        M = cv2.getAffineTransform(p1, p2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(left_point[1]) : int(height_point[1]), int(left_point[0]) : int(new_right_point[0])]

    # 负角度
    elif left_point[1] > right_point[1]:
        new_left_point = [left_point[0], height_point[1]]
        p1 = np.float32([left_point, height_point, right_point])
        p2 = np.float32([new_left_point, height_point, right_point])
        M = cv2.getAffineTransform(p1, p2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(right_point[1]) : int(height_point[1]), int(new_left_point[0]) : int(right_point[0])]

    
    return car_img


def pre_process(img):
    # 图像预处理

    # 灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 模糊
    blur_img = cv2.blur(gray_img, (3, 3))

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间为[26, 34], 蓝色色调区间为[100, 124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    # 图像混合
    mix_img = np.multiply(sobel_img, blue_img)
    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return close_img

# 候选车牌区域应用漫水填充算法
# 作用一: 补全上一步求轮廓时可能存在的轮廓残缺问题
# 作用二: 将非车牌区域排除
def verify_color(rotate_rect, img):
    img_h, img_w = img.shape[:2]
    mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
    # 种子点上下左右4邻域于种子颜色值在[loDiff, upDiff]的被涂成new_value, 也可以设置8邻域
    connectivity = 4
    loDiff = 30
    upDiff = 30
    new_value = 255
    flags = connectivity
    # 考虑当前像素与种子像素之间的差, 如不设置则和邻域像素比较
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= new_value << 8
    # 设置这个标识符,不会填充改变原始图像, 仅仅填充掩模图像mask
    flags |= cv2.FLOODFILL_MASK_ONLY

    # 生成多个随机种子
    seeds = 5000
    # 从所有的随机种子中随机挑选200个有效种子
    valid_seeds = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
    col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x ]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
    row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]
    # 如果以上方法种子点在水平或者垂直方向可以移动的范围很小,则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
        points_row = []
        points_col = []
        for i in range(2):
            p1, p2 = box_points[i], box_points[i+2]
            x_adjust = int(adjust_param * (abs(p1[0] - p2[0])))
            y_adjust = int(adjust_param * (abs(p1[1] - p2[1])))
            if p1[0] <= p2[0]:
                p1[0], p2[0] = p1[0] + x_adjust, p2[0] - x_adjust
            else:
                p1[0], p2[0] = p1[0] - x_adjust, p2[0] + x_adjust
            if p1[1] <= p2[1]:
                p1[1], p2[1] = p1[1] + y_adjust, p2[1] - y_adjust
            else:
                p1[1], p2[1] = p1[1] - y_adjust, p2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(p1[0], p2[0], int(seeds / 2))]
            temp_list_y = [int(y) for y in np.linspace(p1[0], p2[0], int(seeds / 2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0], row_range[1], size=seeds)
        points_col = np.linspace(col_range[0], col_range[1], num=seeds).astype(np.int)
    
    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,s,v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    # 将生成的多个种子依次做漫水填充,理想状态:整个车牌被填充
    flood_img = img.copy()
    seed_count = 0
    for i in range(seeds):
        index = np.random.choice(seeds, 1, replace=False)
        row, col = points_row[index], points_col[index]
        # 随机种子必须是车牌背景色
        if (((h[row, col] > 26 ) & (h[row, col] < 34)) | ((h[row, col] > 100) & (h[row, col] < 124))) & (s[row, col] > 70) & (v[row, col] > 70):
            cv2.floodFill(image, mask, (col, row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            seed_count += 1
            if seed_count >= valid_seeds:
                break
    

    show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
    cv2.imshow('floodfill' + str(show_seed), flood_img)
    cv2.imshow('fllodmask' + str(show_seed), mask)

    # 获取掩模上被填充点的像素点, 并求点集的最小外接矩形
    mask_points = []
    for row in range(1, img_h + 1):
        for col in range(1, img_w + 1):
            if mask[row, col] != 0:
                mask_points.append((col - 1, row - 1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True, mask_rotateRect
    else:
        return False, mask_rotateRect

def find_license(img, pred_img):
    # 寻找图片中的车牌
    license_list = []
    temp_img_1 = img.copy()
    temp_img_2 = img.copy()
    contours, heriachy = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
         cv2.drawContours(temp_img_1, contours, i, (0, 255, 255), 2)
         # 获取轮廓最小的外接矩形，返回值rotate_rect
         rotate_rect = cv2.minAreaRect(contour)
         # 根据矩形面积大小和长宽比判断是否是车牌
         if verify_scale(rotate_rect):
            rec, new_rotate_rect = verify_color(rotate_rect, temp_img_2)
            if rec == False:
                continue
            # 车牌位置矫正
            _license = img_transform(new_rotate_rect, temp_img_2)
            _license_h, _license_w = _license[:2]
            # 将车牌大小调整为cnn网络所需要的大小
            _license = cv2.resize(_license, (_LICENSE_W, _LICENSE_H))
            
            box = cv2.boxPoints(new_rotate_rect)
            for k in range(4):
                n1, n2 = k % 4, (k + 1) % 4
                cv2.line(temp_img_1, (box[n1][0], box[n1][1]), (box[n2][0], box[n2][1]), (255, 0, 0), 2)
            cv2.imshow('opencv_' + str(i), _license)

            license_list.append(_license)
    
    cv2.imshow('contour', temp_img_1)
    return license_list

def horizontal_cut_chars(_license):
    # 车牌号字符串切割
    # char_list中保存左边顶点位置、右边顶点位置、字符宽度
    char_location_list = []
    area_left, area_right, char_left, char_right = 0,0,0,0
    img_w = _license.shape[1]

    # 获取车牌每列边缘像素点个数
    def get_col_sum(img, col):
        sum = 0
        for i in range(img.shape[0]):
            sum+=round(img[i, col] / 255)
        return sum
    
    sum = 0
    for col in range(img_w):
        sum += get_col_sum(_license, col)
    
    # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0
    # 字符宽度限制
    char_width_limit = [round(img_w/12), round(img_w/5)]
    is_char_flag = False

    for i in range(img_w):
        col_value = get_col_sum(_license, i)
        if col_value > col_limit:
            if is_char_flag == False:
                area_right = round((i + char_right) / 2)
                area_width = area_right - area_left
                char_width = char_right - char_left
                if area_width > char_width_limit[0] and area_width < char_width_limit[1]:
                    char_location_list.append((area_left, area_right, char_width))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i -1
                is_char_flag = False
    
    # 结束最后的未完成的字符串的分割
    if area_right < char_left:
        area_right = char_right = img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if area_width > char_width_limit[0] and area_width < char_width_limit[1]:
            char_location_list.append((area_left, area_right, char_width))
    return char_location_list


def get_char(car_license):
    img_h, img_w = car_license.shape[:2]
    # 水平投影长度列表
    h_proj_list = []
    h_len = v_len = 0
    # 水平投影记索引
    h_start_index = h_end_index = 0
    # 车牌在水平方向的轮廓长度少于20%或者多于80%过滤掉
    h_proj_limit = [0.2, 0.8]
    chars = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止有一段
    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        count = 0
        for col in range(img_w):
            if car_license[row, col] == 255:
                count += 1
        h_count[row] = count
        if count / img_w < h_proj_limit[0] or count / img_w > h_proj_limit[1]:
            if h_len != 0:
                h_end_index = row - 1
                h_proj_list.append((h_start_index, h_end_index))
                h_len = 0
            continue
        if count > 0:
            if h_len == 0:
                h_start_index = row
                h_len = 1
            else:
                h_len += 1
        else:
            if h_len > 0:
                h_end_index = row - 1
                h_proj_list.append((h_start_index, h_end_index))
                h_len = 0
    
    # 结束最后的水平投影长度累加
    if h_len != 0:
        h_end_index = img_h - 1
        h_proj_list.append((h_start_index, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_max_index = h_max_height = 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_max_height < (end- start):
            h_max_height = end - start
            h_max_index = i
    if h_max_height / img_h < 0.5:
        return chars
    chars_top = h_proj_list[h_max_index][0]
    chars_bottom = h_proj_list[h_max_index][1]

    licenses = car_license[chars_top: chars_bottom + 1, :]
    # cv2.imwrite('./results/cars/' + str(car_pic_name) + '.jpg', car_license)
    # cv2.imwrite('./results/licenses/' + str(car_pic_name) + '_license.jpg', licenses)

    char_location_list = horizontal_cut_chars(licenses)

    for i, location in enumerate(char_location_list):
        char_img = car_license[chars_top : chars_bottom + 1, location[0] : location[1]]
        char_img = cv2.resize(char_img, (_CHAR_W, _CHAR_H))
        chars.append(char_img)
    return chars

def extract_char(car_license):
    gray_license = cv2.cvtColor(car_license, cv2.COLOR_BGR2GRAY)
    rec, binary_license = cv2.threshold(gray_license, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    char_img_list = get_char(binary_license)
    return char_img_list

def cnn_select_license(license_list, model_path):
    if len(license_list) == 0:
        return False, license_list
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with sess.as_default():
        with sess.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            x_place = graph.get_tensor_by_name('x_place:0')
            keep_place = graph.get_tensor_by_name('keep_place:0')
            out = graph.get_tensor_by_name('out_put:0')

            input_x = np.array(license_list)
            net_outs = tf.nn.softmax(out)
            # 预测结果
            preds = tf.argmax(net_outs, 1)
            # 预测结果的概率值
            probs = tf.reduce_max(net_outs, reduction_indices=[1])
            pred_list, prob_list = sess.run([preds, probs], feed_dict={x_place:input_x, keep_place:1.0})

            # 选出最大概率值的车牌
            result_index, result_prob = -1, 0
            for i, pred in enumerate(pred_list):
                if pred == 1  and prob_list[i] > result_prob:
                    result_index = i
                    result_prob = prob_list[i]
            if result_index == -1:
                return False, license_list[0]
            else:
                return True, license_list[result_index]

def cnn_recognize_char(img_list, model_path):
    g = tf.Graph()
    sess = tf.Session(graph=g)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess.as_default():
        with sess.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            x_place = graph.get_tensor_by_name('x_place:0')
            keep_place = graph.get_tensor_by_name('keep_place:0')
            out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            outs = tf.nn.softmax(out)
            preds = tf.argmax(outs, 1)
            preds = sess.run(preds, feed_dict={x_place:data, keep_place:1.0})

            for i in preds:
                text_list.append(license_char[i])
            return text_list

if __name__ == '__main__':
    current_path = sys.path[0]
    _LICENSE_W, _LICENSE_H = 136, 36
    _CHAR_W, _CHAR_H = 20, 20

    license_model_path = os.path.join(current_path, args.license_model)
    char_model_path = os.path.join(current_path, args.char_model)
    image_path = os.path.join(current_path, 'images/', args.image)
    result_path = os.path.join(current_path, 'results/', ('_'.join(args.image.split('.'))) + '.txt')

    image =  cv2.imread(image_path)

    # 图像预处理
    pred_img = pre_process(image)

    # 车牌定位
    license_list = find_license(image, pred_img)

    # 使用CNN网络进行车牌过滤
    res, _license = cnn_select_license(license_list, license_model_path)
    if res == False:
        print('未检测到车牌')
        sys.exit(-1)
    cv2.imshow('license selected by CNN', _license)

    # 提取字符
    char_list = extract_char(_license)

    # 使用CNN网络识别字符
    text = cnn_recognize_char(char_list, char_model_path)
    print(text)
    
    # 使用utf-8编码解决中文写入问题
    result = open(result_path, 'w', encoding='utf-8')
    result.write(''.join(text))

    cv2.waitKey()
    
