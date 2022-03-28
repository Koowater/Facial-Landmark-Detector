import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy

focal_loss = BinaryFocalCrossentropy()

def total_loss(y_true, y_pred):
    """
    y_true.shape = (batch, NumAnc, 5))
    """
    gt_classes = y_true[:,:,:1] # (batch_size, NumofAnchors, 1)
    gt_boxes = y_true[:,:,1:5] # (batch_size, NumofAnchors, 4)
    
    pr_classes = y_pred[:,:,:1]
    pr_boxes = y_pred[:,:,1:5]
    
    batch_size  = tf.shape(gt_classes)[0]
    anchor_num = tf.shape(gt_classes)[1]
    total_num = tf.ones([batch_size], dtype=tf.int64) * tf.cast(anchor_num, tf.int64)
    
    # ======= Conf loss =======
    positives_num = tf.compat.v1.count_nonzero(gt_classes[:,:,0], axis=1)
    negatives_num = total_num - positives_num

    positives_num_safe = tf.where(tf.equal(positives_num, 0),
                              tf.ones([batch_size])*10e-15,
                              tf.cast(positives_num, tf.float32))

    negatives_mask = tf.equal(gt_classes[:,:,0], 0)
    # Shape: (batch_size, num_anchors)
    positives_mask = tf.logical_not(negatives_mask)
    
    #Focal loss. 
    fc_loss = focal_loss(gt_classes, pr_classes)
    # Shape: (batch_size, num_anchors)
    positives = tf.where(positives_mask, fc_loss, tf.zeros_like(fc_loss))
    # Shape: (batch_size)
    positives_sum = tf.reduce_sum(positives, axis=-1)
    
    negatives = tf.where(negatives_mask, fc_loss, tf.zeros_like(fc_loss))
    # Shape: (batch_size, num_anchors)
    negatives_top = tf.nn.top_k(negatives, anchor_num)[0]
    # Maximum number of negatives to keep per sample - we keep at most
    # 3 times as many as we have positive anchors in the sample
    negatives_num_max = tf.minimum(negatives_num, 3 * positives_num)
    
    negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)
    rng = tf.range(0, anchor_num, 1)
    range_row = tf.compat.v1.to_int64(tf.expand_dims(rng, 0))
    negatives_max_mask = tf.less(range_row, negatives_num_max_t)
    # Shape: (batch_size, num_anchors)
    negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))
    negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)
    confidence_loss_sum = tf.add(positives_sum, negatives_max_sum)
    confidence_loss_sum = tf.cast(confidence_loss_sum, tf.float32)
    confidence_loss = tf.where(tf.equal(positives_num, 0),
                               tf.zeros([batch_size]),
                               tf.math.divide(confidence_loss_sum, positives_num_safe))

    # Bounding box regresssion Loss
    
    box_loss = smooth_l1(gt_boxes, pr_boxes)
    
    positive_box = tf.where(positives_mask, box_loss,
                             tf.zeros_like(box_loss))
    positive_box_loss_sum = tf.cast(tf.reduce_sum(positive_box, axis=-1), tf.float32)

    positive_box_loss = tf.where(tf.equal(positives_num, 0),
                                 tf.zeros([batch_size]),
                                 tf.math.divide(positive_box_loss_sum, positives_num_safe))

    class_box_loss = tf.math.add(tf.math.multiply(confidence_loss, 1.), positive_box_loss)

    return class_box_loss

def conf_metric(y_true, y_pred):
    """
    y_true.shape = (batch, NumAnc, 5))
    """
    gt_classes = y_true[:,:,:1] # (batch_size, NumofAnchors, 1)
    
    pr_classes = y_pred[:,:,:1]
    
    batch_size  = tf.shape(gt_classes)[0]
    anchor_num = tf.shape(gt_classes)[1]
    total_num = tf.ones([batch_size], dtype=tf.int64) * tf.cast(anchor_num, tf.int64)
    
    # ======= Conf loss =======
    positives_num = tf.compat.v1.count_nonzero(gt_classes[:,:,0], axis=1)
    negatives_num = total_num - positives_num

    positives_num_safe = tf.where(tf.equal(positives_num, 0),
                              tf.ones([batch_size])*10e-15,
                              tf.cast(positives_num, tf.float32))

    negatives_mask = tf.equal(gt_classes[:,:,0], 0)
    # Shape: (batch_size, num_anchors)
    positives_mask = tf.logical_not(negatives_mask)
    
    #Focal loss. 
    fc_loss = focal_loss(gt_classes, pr_classes)
    # Shape: (batch_size, num_anchors)
    positives = tf.where(positives_mask, fc_loss, tf.zeros_like(fc_loss))
    # Shape: (batch_size)
    positives_sum = tf.reduce_sum(positives, axis=-1)
    
    negatives = tf.where(negatives_mask, fc_loss, tf.zeros_like(fc_loss))
    # Shape: (batch_size, num_anchors)
    negatives_top = tf.nn.top_k(negatives, anchor_num)[0]
    # Maximum number of negatives to keep per sample - we keep at most
    # 3 times as many as we have positive anchors in the sample
    negatives_num_max = tf.minimum(negatives_num, 3 * positives_num)
    
    negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)
    rng = tf.range(0, anchor_num, 1)
    range_row = tf.compat.v1.to_int64(tf.expand_dims(rng, 0))
    negatives_max_mask = tf.less(range_row, negatives_num_max_t)
    # Shape: (batch_size, num_anchors)
    negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))
    negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)
    confidence_loss_sum = tf.add(positives_sum, negatives_max_sum)
    confidence_loss_sum = tf.cast(confidence_loss_sum, tf.float32)
    confidence_loss = tf.where(tf.equal(positives_num, 0),
                               tf.zeros([batch_size]),
                               tf.math.divide(confidence_loss_sum, positives_num_safe))

    return tf.math.multiply(confidence_loss, 1.)

def box_metric(y_true, y_pred):
    """
    y_true.shape = (batch, NumAnc, 5))
    """
    gt_classes = y_true[:,:,:1] # (batch_size, NumofAnchors, 1)
    gt_boxes = y_true[:,:,1:5] # (batch_size, NumofAnchors, 4)
    
    pr_boxes = y_pred[:,:,1:5]
    
    batch_size  = tf.shape(gt_classes)[0]

    # ======= Conf loss =======
    positives_num = tf.compat.v1.count_nonzero(gt_classes[:,:,0], axis=1)

    positives_num_safe = tf.where(tf.equal(positives_num, 0),
                              tf.ones([batch_size])*10e-15,
                              tf.cast(positives_num, tf.float32))

    negatives_mask = tf.equal(gt_classes[:,:,0], 0)
    # Shape: (batch_size, num_anchors)
    positives_mask = tf.logical_not(negatives_mask)

    # Bounding box regresssion Loss
    box_loss = smooth_l1(gt_boxes, pr_boxes)
    
    positive_box = tf.where(positives_mask, box_loss,
                             tf.zeros_like(box_loss))
    positive_box_loss_sum = tf.cast(tf.reduce_sum(positive_box, axis=-1), tf.float32)

    positive_box_loss = tf.where(tf.equal(positives_num, 0),
                                 tf.zeros([batch_size]),
                                 tf.math.divide(positive_box_loss_sum, positives_num_safe))

    return positive_box_loss

def smooth_l1(x):

    def func1():
        return x**2 * 0.5

    def func2():
        return tf.abs(x) - tf.constant(0.5)

    # tf.cond: pred(cond의 첫 번째 파라메터)가 참이면 true_fn, 아니면 fase_fn을 반환한다.
    # tf.less: x < y를 만족하면 True를 반환한다.
    # f(x): 입력값의 절대값이 1 미만이면 func1, 그렇지 않으면 func2를 반환한다.
    def f(x): return tf.cond(tf.less(tf.abs(x), tf.constant(1.0)), func1, func2)

    return tf.map_fn(f, x)

def smooth_l1_loss(true, pred):

    """
    compute smooth_l1 loss 
    g : ground truth
    p = prediction
    hat{g}(x) = (g_x - p_x) / p_w
    hat{g}(y) = (g_y - p_y) / p_hi
    hat{g}(w) = log(g_w / p_w)
    hat{g}(h) = log(g_h / p_h)
    smooth_l1_loss = reduce_mean(smooth_l1(g - hat{g}))
    """

    face_label = true[:, :, :1]

    gxs = true[:, :, 1:2]
    gys = true[:, :, 2:3]
    gws = true[:, :, 3:4]
    ghs = true[:, :, 4:5]

    pxs = pred[:, :, 1:2]
    pys = pred[:, :, 2:3]
    pws = pred[:, :, 3:4]
    phs = pred[:, :, 4:5]

    logitx = (gxs - pxs) / pws
    logity = (gys - pys) / phs
    logitw = tf.math.log(gws / pws)
    logith = tf.math.log(ghs / phs)

    lossx = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gxs - logitx, (-1, 896)))
    lossy = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gys - logity, (-1, 896)))
    lossw = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gws - logitw, (-1, 896)))
    lossh = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(ghs - logith, (-1, 896)))

    x_sum = tf.reduce_sum(lossx)
    y_sum = tf.reduce_sum(lossy)
    w_sum = tf.reduce_sum(lossw)
    h_sum = tf.reduce_sum(lossh)

    loss = tf.stack((x_sum, y_sum, w_sum, h_sum))

    return tf.reduce_mean(loss)