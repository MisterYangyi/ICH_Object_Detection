import math

from tqdm import tqdm

from utils import utils_map as map
from matplotlib import pyplot as plt
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

from utils.utils_map import file_lines_to_list, voc_ap


def draw_PR_curve(dfs, path, iou, hosptal=""):
    classnames = ["IVH", "IPH", "SAH", "SDH", "EDH"]
    colors = [["#27736E"], ["#299D91"], ["#8BB17B"], ["#E8C56A"], ["#F2A461"]]
    plt.rc('font', family='Arial', size=15)
    labels = []
    fig = plt.figure(figsize=(7, 6), dpi=250)
    ax1 = plt.subplot(111)
    ax1.spines['bottom'].set_linewidth('1.2')
    ax1.spines['top'].set_linewidth('1.2')
    ax1.spines['left'].set_linewidth('1.2')
    ax1.spines['right'].set_linewidth('1.2')
    data_confidence = []
    aps = []
    for name in classnames:
        data = dfs[name]
        data_r = data.copy()
        data_r.drop_duplicates(subset=["F1"], keep='first', inplace=True)
        temp = data[data["F1"] == data_r["F1"].max()].values.tolist()[0]
        data_confidence.append([name] + temp)
        data.drop_duplicates(subset=["Recall"], keep='first', inplace=True)
        data = data[data["Precision"] != 0]
        precision = data["Precision"].values.tolist()
        recall = data["Recall"].values.tolist()
        precision1 = precision.copy()
        recall1 = recall.copy()
        ap, mrec, mprec = map.voc_ap(recall, precision)
        aps.append(ap)
        label = '%s (AP=%0.3f)' % (name, ap)
        lines = []
        labels.append(label)
        x = [0.0] + recall1 + [recall1[-1]]
        y = [precision1[0]] + precision1 + [0]
        ax1.plot(x, y, linestyle='-', lw=2,
                 color=colors[classnames.index(name)][0],
                 label=label, zorder=1,
                 alpha=1.0)
    plt.xlabel('Recall', fontdict={"size": 20})
    plt.ylabel('Precision', fontdict={"size": 20})
    ax1.set_xlim([0.0, 1.02])
    ax1.set_ylim([0, 1.02])
    plt.rc('legend', fontsize='large')
    # plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, )
    plt.legend()
    plt.tick_params(labelsize=20)
    plt.show()
    fig.savefig(os.path.join(path, "%s_%s_Pr曲线.pdf" % (iou, hosptal)))
    df_conf = pd.DataFrame(data=data_confidence, columns=["name", "confidence", "recall", "precision", "F1"])
    df_conf["mAP"] = aps
    df_conf.to_csv(os.path.join(path, "%s_%s_confidence.csv" % (iou, hosptal)), index=False)
    return df_conf


def wilson(a, b):
    Q1 = 2 * a + 3.84
    Q2 = 1.96 * math.sqrt(3.84 + ((4 * a * b) / (a + b)))
    Q3 = 2 * (a + b) + 7.68
    return np.round(np.array([(Q1 - Q2) / Q3, (Q1 + Q2) / Q3]), 3)


def get_xml_info(txt_file):
    lines_list = file_lines_to_list(txt_file)
    objs = []
    for line in lines_list:
        if "difficult" in line:
            class_name, left, top, right, bottom, _difficult = line.split()
        else:
            class_name, left, top, right, bottom = line.split()
        objs.append([class_name, [int(top), int(left), int(bottom), int(right)]])
    return objs


def get_xml_model_predict(txt_file, df):
    lines_list = file_lines_to_list(txt_file)
    objs = []
    for line in lines_list:
        class_name, confidence, left, top, right, bottom = line.split()
        confidence_th = df[df["name"] == class_name].values.tolist()[0][1]
        if float(confidence) >= confidence_th:
            objs.append([class_name, [int(top), int(left), int(bottom), int(right)]])
    return objs


def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点

    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def get_tp_fp_fn(gt_objs, pre_objs, classes, iou_):
    result = []
    for cls in classes:
        tp = 0
        gt_cls = []
        pre_cls = []
        for gt_obj in gt_objs:
            if gt_obj[0] == cls:
                gt_cls.append(gt_obj)
        for pre_obj in pre_objs:
            if pre_obj[0] == cls:
                pre_cls.append(pre_obj)
        pre = len(pre_cls)
        gt = len(gt_cls)
        for gt_cl in gt_cls:
            ious = []
            for pre_cl in pre_cls:
                ioua = iou(gt_cl[1], pre_cl[1])
                ious.append(ioua)
            if len(ious) != 0:
                iou_max = np.array(ious).max()
                if iou_max >= iou_:
                    index = ious.index(iou_max)
                    pre_cls.remove(pre_cls[index])
                    # print(pre_cls, iou_max)
                    tp += 1
        fp = pre - tp
        fn = gt - tp
        result.append(tp)
        result.append(fp)
        result.append(fn)
    return result


def get_tpfpfn(conf_df, path, iou):
    classes = ["IVH", "IPH", "SAH", "SDH", "EDH"]
    pre_dir = path + "/detection-results"
    gt_dir = path + "/ground-truth"
    sample = os.listdir(gt_dir)
    results = []
    for xml in tqdm(sample):
        pre_objs = get_xml_model_predict(os.path.join(pre_dir, xml), conf_df)
        gt_objs = get_xml_info(os.path.join(gt_dir, xml))
        result = get_tp_fp_fn(gt_objs, pre_objs, classes, iou_=iou)
        n = 0
        count = 0
        for re in result:
            if not n % 3 == 0:
                count = count + re
            n += 1
        results.append([xml.replace(".xml", "")] + result)
    columns = ["xml_name"]
    for cls in classes:
        columns.append(cls + "TP")
        columns.append(cls + "FP")
        columns.append(cls + "FN")
    df = pd.DataFrame(data=results, columns=columns)
    df.to_csv(os.path.join(path, "%.2f_tpfpfn.csv" % (iou)))

    all_precision = []
    all_recall = []
    all_f1 = []
    precisions = []
    recalls = []
    f1s = []
    dic = {}
    for i in range(len(classes)):
        name1 = classes[i] + 'TP'
        name2 = classes[i] + 'FP'
        name3 = classes[i] + 'FN'

        TP = np.sum(df[name1].values.tolist())
        FP = np.sum(df[name2].values.tolist())
        FN = np.sum(df[name3].values.tolist())

        print(classes[i] + '(' + classes[i] + ')')

        Precision = TP / (TP + FP)
        [a, b] = wilson(TP, FP)
        c = "%.3f\n(%.3f-%.3f)" % (Precision, a, b)
        precisions.append(c)
        print("Precision：\t\t", c)

        Recall = TP / (TP + FN)
        [a, b] = wilson(TP, FN)
        c = "%.3f\n(%.3f-%.3f)" % (Recall, a, b)
        recalls.append(c)
        print("Recall：\t\t", c)

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        c = "%.3f" % F1
        f1s.append(c)
        print("F1_score：\t\t", round(F1, 3))

        print('---------------------------------------------')

        all_precision.append(Precision)
        all_recall.append(Recall)
        all_f1.append(F1)

    print('Mean_Precision:', round(np.mean(all_precision), 3))
    print('Mean_Recall:', round(np.mean(all_recall), 3))
    print('Mean_f1:', round(np.mean(all_f1), 3))
    dic["Classes"] = classes + ["Mean"]
    dic["Precision"] = precisions + [round(np.mean(all_precision), 3)]
    dic["Recall"] = recalls + [round(np.mean(all_recall), 3)]
    dic["F1"] = f1s + [round(np.mean(all_f1), 3)]
    dd = pd.DataFrame(data=dic)
    dd.to_excel(os.path.join(path, "%.2f_result.xlsx" % (iou)), index=False)


def get_tpfpfn_man(doctor_name, path, iou):
    classes = ["IVH", "IPH", "SAH", "SDH", "EDH"]
    pre_dir = path + "/" + doctor_name
    gt_dir = path + "/gt"
    sample = os.listdir(gt_dir)
    results = []
    for xml in tqdm(sample):
        pre_objs = get_xml_info(os.path.join(pre_dir, xml))
        gt_objs = get_xml_info(os.path.join(gt_dir, xml))
        result = get_tp_fp_fn(gt_objs, pre_objs, classes, iou_=iou)
        n = 0
        count = 0
        for re in result:
            if not n % 3 == 0:
                count = count + re
            n += 1
        results.append([xml.replace(".xml", "")] + result)
    columns = ["xml_name"]
    for cls in classes:
        columns.append(cls + "TP")
        columns.append(cls + "FP")
        columns.append(cls + "FN")
    df = pd.DataFrame(data=results, columns=columns)
    df.to_csv(os.path.join(path, "%s_%.2f_tpfpfn.csv" % (doctor_name, iou)))

    all_precision = []
    all_recall = []
    all_f1 = []
    precisions = []
    recalls = []
    f1s = []
    dic = {}
    for i in range(len(classes)):
        name1 = classes[i] + 'TP'
        name2 = classes[i] + 'FP'
        name3 = classes[i] + 'FN'

        TP = np.sum(df[name1].values.tolist())
        FP = np.sum(df[name2].values.tolist())
        FN = np.sum(df[name3].values.tolist())

        print(classes[i] + '(' + classes[i] + ')')

        Precision = TP / (TP + FP)
        [a, b] = wilson(TP, FP)
        c = "%.3f" % (Precision)
        precisions.append(c)
        print("Precision：\t\t", c)

        Recall = TP / (TP + FN)
        [a, b] = wilson(TP, FN)
        c = "%.3f" % (Recall)
        recalls.append(c)
        print("Recall：\t\t", c)

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        c = "%.3f" % F1
        f1s.append(c)
        print("F1_score：\t\t", round(F1, 3))

        print('---------------------------------------------')
        all_precision.append(Precision)
        all_recall.append(Recall)
        all_f1.append(F1)

    print('Mean_Precision:', round(np.mean(all_precision), 3))
    print('Mean_Recall:', round(np.mean(all_recall), 3))
    print('Mean_f1:', round(np.mean(all_f1), 3))
    dic["Classes"] = classes
    dic["Precision"] = precisions
    dic["Recall"] = recalls
    dic["F1"] = f1s
    dd = pd.DataFrame(data=dic)
    dd.to_excel(os.path.join(path, "0001_%s_%.2f_result.xlsx" % (doctor_name, iou)), index=False)


def draw_ap_by_noe(dfs, iou, path, hosptal=""):
    plt.rc('font', family='Arial', size=20)
    colors = [["#27736E"], ["#299D91"], ["#8BB17B"], ["#E8C56A"], ["#F2A461"]]
    if iou == 0.1:
        color = colors[2]
    elif iou == 0.3:
        color = colors[1]
    else:
        color = colors[0]
    fig = plt.figure(figsize=(18, 12), dpi=300)

    names = [
        "IVH", "IPH", "SAH", "SDH", "EDH"
    ]
    doctor_names = ["FWJ", "PQ", "WF", "WXQ"]
    for name in names:
        plt.subplot(2, 3, names.index(name) + 1)
        data = dfs[name]
        data.drop_duplicates(subset=["Confidence"], keep='last', inplace=True)
        data_r = data.copy()
        data_r.drop_duplicates(subset=["F1"], keep='last', inplace=True)
        temp = data[data["F1"] == data_r["F1"].max()].values.tolist()[0]
        data.drop_duplicates(subset=["Recall"], keep='first', inplace=True)
        data = data[data["Precision"] != 0]
        precision = data["Precision"].values.tolist()
        recall = data["Recall"].values.tolist()
        precision1 = precision.copy()
        recall1 = recall.copy()
        ap, mrec, mprec = voc_ap(recall, precision)
        doctor_data = pd.read_excel(path + "/doctor_0.50.xlsx",
                                    sheet_name="%.1f" % iou)
        l, m, n, o = 0, 0, 0, 0
        plt.plot([0.0] + recall1 + [recall1[-1]], [precision1[0]] + precision1 + [0], linestyle='-', lw=4,
                 color="#4169E1",
                 label='AP=%0.3f' % (ap), zorder=1,
                 alpha=1.0)
        colors = [["#27736E"], ["#299D91"], ["#8BB17B"], ["#E8C56A"], ["#F2A461"]]
        for doctor_name in doctor_names:
            doc_da = doctor_data[doctor_data["doctor_name"] == doctor_name]
            da = doc_da[doc_da["Classes"] == name].values.tolist()[0]
            if doctor_name in ["FWJ", ]:  # 高级医生
                l += 1
                # edgecolors = "black"
                plt.scatter(da[3], da[2], marker="^", alpha=1, zorder=2, s=300, color="#0ca84c")
            elif doctor_name in ["PQ"]:  # 中级
                m += 1
                plt.scatter(da[3], da[2], marker="^", alpha=1, s=300, color="#8854a0")
            elif doctor_name in ["WF"]:  # 初级
                n += 1
                plt.scatter(da[3], da[2], marker="^", alpha=1, s=300, color="#ffcb4b")
            elif doctor_name in ["WXQ"]:  # 初级
                o += 1
                plt.scatter(da[3], da[2], marker="^", alpha=1, s=300, color="#ec7c20")

            plt.scatter(temp[-3], temp[-2], s=200, color="#ca3441", marker="o", alpha=1,
                        zorder=2)  # 画点
        plt.title('%s' % name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        axes = plt.gca()
        plt.legend(loc="lower left")
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])

    plt.show()
    fig.savefig(os.path.join(path, "%0.1f_%s_Pr曲线2.pdf" % (iou, hosptal)))


if __name__ == '__main__':
    # path = "J:\RSAN\测试集\潍坊\TXT"
    # get_tpfpfn_man("04", path, 0.7)
    # exit(0)

    # path = "/home/Yang/Project/ICH/ICH_YOLOV7/27_runs/01_rsna_test_120830"
    path = "/home/Yang/Project/ICH/YOLOV8.1/runs/detect/trian_Concat_BiFPN/PD2_Test"
    ious = [0.1, 0.3, 0.5, 0.7]
    # ious = [0.9]

    # exit(0)
    for iou1 in ious:
        map1, dfs = map.get_map(iou1, True, score_threhold=0.1,
                                path=path)
        # draw_ap_by_noe(dfs, iou1, path, "")
        #
        conf_df = draw_PR_curve(dfs, path, "%.3f" % iou1, "cq500")
        # #
        get_tpfpfn(conf_df, path, iou1)
