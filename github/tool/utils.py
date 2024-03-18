import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


def OA(pre_classes, gt_classes):
    return torch.sum((pre_classes) == (gt_classes)).float()/len(pre_classes)


def Load_Weight_FordataParallel(state_dict, need_dataparallel=0):
        if_dataparallel = 1
        for k, v in state_dict.items():
            name = k[:6]
            if name != "module":
                if_dataparallel = 0
        if need_dataparallel == 1:
            if if_dataparallel == 1:
                return state_dict
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = "module."+k 
                    new_state_dict[name] = v 
                return new_state_dict
        else:
            if if_dataparallel == 0:
                return state_dict
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] 
                    new_state_dict[name] = v 
                return new_state_dict 
        
def test_one_model(data_loader,model,Log_path,epoch=0,best_acc=0, save_name=None):
    model.eval()
    with torch.no_grad():
        Accuracies = []
        Right = 0
        Sum = 0

        TP = 0
        FP = 0
        FN = 0

        for i,(i1,i2,label,file_name,mask) in enumerate(tqdm(data_loader)):
            i1,i2,label = i1.cuda(),i2.cuda(),label.cuda()

            prediction = model(i1,i2)
            prediction = torch.max(prediction,dim=1)[1]

            Right += torch.sum(prediction == label)
            Sum += torch.sum(label>-1)

            impred = prediction
            imlabel = label

            accuracy = OA(prediction.view(-1),label.view(-1))
            Accuracies.append(float(accuracy))

            #Precision,recall,Iou
            numclass = 1
            TP +=  int(torch.sum(impred * (impred == imlabel)))
            FP += int(torch.sum(impred * (impred != imlabel)))
            FN += int(torch.sum(imlabel * (impred != imlabel)))

        Average_accuracy = np.mean(Accuracies)
        Overrall_accuracy = float(Right/Sum)
        Iou = TP/(TP+FP+FN)
        try:
            Percison = TP/(TP+FP)
        except:
            Percison = 0
        Recall = TP/(TP+FN)
        try:
            F1 = (2*Percison*Recall)/(Percison+Recall)
        except:
            F1 = 0

        Average_accuracy = round(Average_accuracy,4)
        Overrall_accuracy = round(Overrall_accuracy,4)
        Iou = round(Iou,4)
        Percison = round(Percison,4)
        Recall = round(Recall,4)
        F1 = round(F1,4)

        print('AA: \t\t',Average_accuracy)
        print('OA:\t\t',Overrall_accuracy)
        print('Iou:\t\t',Iou)
        print('Percison:\t',Percison)
        print('Recall:\t\t',Recall)
        print('F1\t\t',F1)
    
    f = open(Log_path+'Metric_recording.txt','a')
    f.write('Epoch:'+str(epoch)+' -- C_F1:'+str(round(F1,4))+' -- B_F1:'+str(round(best_acc,4)))
    f.write(' -- Percison:'+str(Percison)+' -- Recall:'+str(Recall)+' -- F1:'+str(F1)+' -- Iou:'+str(Iou)+' -- OA:'+str(Overrall_accuracy)+'\n')
    f.close()
    
    if F1 >= best_acc:
        best_acc = F1
        state_dict = Load_Weight_FordataParallel(model.state_dict(),need_dataparallel=0)
        if save_name is None:
            torch.save(state_dict, Log_path+'Best_model.pth')
        else:
            torch.save(state_dict, Log_path+save_name+'.pth')
    return best_acc
    

def test_tea_model(data_loader,tea_model, std_models, Log_path,epoch=0,best_acc=0):
    tea_model.eval()
    for model in std_models:
        model.eval()
    with torch.no_grad():
        Accuracies = []
        Right = 0
        Sum = 0

        TP = 0
        FP = 0
        FN = 0

        for i,(i1,i2,label,file_name,mask) in enumerate(tqdm(data_loader)):
            i1,i2,label = i1.cuda(),i2.cuda(),label.cuda()

            prediction = tea_model(i1,i2)
            prediction = torch.max(prediction,dim=1)[1]

            Right += torch.sum(prediction == label)
            Sum += torch.sum(label>-1)

            impred = prediction
            imlabel = label

            accuracy = OA(prediction.view(-1),label.view(-1))
            Accuracies.append(float(accuracy))

            #Precision,recall,Iou
            numclass = 1
            TP +=  int(torch.sum(impred * (impred == imlabel)))
            FP += int(torch.sum(impred * (impred != imlabel)))
            FN += int(torch.sum(imlabel * (impred != imlabel)))

        Average_accuracy = np.mean(Accuracies)
        Overrall_accuracy = float(Right/Sum)
        Iou = TP/(TP+FP+FN)
        try:
            Percison = TP/(TP+FP)
        except:
            Percison = 0
        Recall = TP/(TP+FN)
        try:
            F1 = (2*Percison*Recall)/(Percison+Recall)
        except:
            F1 = 0

        Average_accuracy = round(Average_accuracy,4)
        Overrall_accuracy = round(Overrall_accuracy,4)
        Iou = round(Iou,4)
        Percison = round(Percison,4)
        Recall = round(Recall,4)
        F1 = round(F1,4)

        print('AA: \t\t',Average_accuracy)
        print('OA:\t\t',Overrall_accuracy)
        print('Iou:\t\t',Iou)
        print('Percison:\t',Percison)
        print('Recall:\t\t',Recall)
        print('F1\t\t',F1)
    
    f = open(Log_path+'Metric_recording.txt','a')
    f.write('Epoch:'+str(epoch)+' -- C_F1:'+str(round(F1,4))+' -- B_F1:'+str(round(best_acc,4)))
    f.write(' -- Percison:'+str(Percison)+' -- Recall:'+str(Recall)+' -- F1:'+str(F1)+' -- Iou:'+str(Iou)+' -- OA:'+str(Overrall_accuracy)+'\n')
    f.close()
    
    if F1 >= best_acc:
        best_acc = F1
        state_dict = Load_Weight_FordataParallel(tea_model.state_dict(),need_dataparallel=0)
        torch.save(state_dict, Log_path+'Best_tea_model.pth')
        for i,model in enumerate(std_models):
            state_dict = Load_Weight_FordataParallel(model.state_dict(),need_dataparallel=0)
            torch.save(state_dict, Log_path+'Best_stu_model'+str(i+1)+'.pth')
    return best_acc