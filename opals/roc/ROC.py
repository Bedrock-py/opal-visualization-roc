#****************************************************************
#
# Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from bedrock.visualization.utils import *
from bedrock.visualization.colors import brews
import vincent, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp



def get_classname():
    return 'ROC'
    
class ROC(Visualization):
    def __init__(self):
        super(ROC, self).__init__()
        self.inputs = ['assignments.csv', 'truth_labels.csv']
        self.parameters = []
        self.parameters_spec = [
                     { "name" : "Type", "attrname" : "type", "value" : "cumulative", "type" : "select", "options": ["cumulative","individual"] },
                     { "name" : "Colors", "attrname" : "color", "value" : "default", "type" : "select", "options": ["default","bright","pastel"] }
        ]
        self.name = 'ROC'
        self.description = ''

    def initialize(self, inputs):
        self.results = np.genfromtxt(inputs['assignments.csv']['rootdir'] + 'assignments.csv', delimiter=',')
        self.truth = np.genfromtxt(inputs['truth_labels.csv']['rootdir'] + 'truth_labels.csv', delimiter=',')
        self.truth = label_binarize(self.truth, classes=np.unique(self.truth))
        self.results = label_binarize(self.results, classes=np.unique(self.results))


    def create(self):
        n_classes = self.truth.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.truth[:, i], self.results[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(self.truth.ravel(), self.results.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        # data = {}
        # y_label = 'Cumulative'
        # data['x'] = fpr[2]
        # data[y_label] = tpr[2]
        # data['k'] = [0,1]

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        data = {}
        data['x'] = [0,1]
        data['k--'] = [0,1]
        # data["macro"] = mean_tpr

        vis = vincent.Line(data, iter_idx='x')

        str_fpr = {}
        for key in fpr.keys():
            str_fpr[str(key)] = fpr[key]

        for key in fpr.keys():
            d = {}
            d[str(key)] = tpr[key]
            d['x'] = fpr[key]
            temp = vincent.Line(d, iter_idx='x')
            temp.data[0].name = str(key)
            temp.legend(title="")
            vis.data.append(temp.data[0])

        legend_content = []
        for key in fpr.keys():

            from_ = vincent.MarkRef(
                        data=str(key),
                        transform=[vincent.Transform(type='facet', keys=['data.col'])])
            
            enter_props = vincent.PropertySet(
                        x=vincent.ValueRef(scale='x', field="data.idx"),
                        y=vincent.ValueRef(scale='y', field="data.val"),
                        stroke=vincent.ValueRef(value=brews['Category10'][int(key)]),
                        stroke_width=vincent.ValueRef(value=2)
                        )

            legend_content.append('Class_' + str(key))
            marks = [vincent.Mark(type='line',
                                  properties=vincent.MarkProperties(enter=enter_props))]
            mark_group = vincent.Mark(type='group', from_=from_, marks=marks)
            vis.marks.append(mark_group)

        vis.scales += [
                vincent.Scale(name='color', range='category10', type='ordinal',
                      domain=str_fpr.keys())]


        vis.legends.append(vincent.legends.Legend(title="", fill='color', offset=0,
                                   properties=vincent.legends.LegendProperties(), values=legend_content))

        vis.axis_titles(x='False Positive Rate', y='True Positive Rate')        
        json_content = vis.to_json()
        data = json.loads(json_content)
        json_content = json.dumps(data)

        vis_id = 'vis_' + get_new_id()
        script = '<script> spec =' + json_content + ';vg.parse.spec(spec, function(chart) { chart({el:"#' + vis_id + '"}).update(); });</script>'
        
        return {'data':script,'type':'default', 'id': vis_id, 'title': 'ROC Curve'}
