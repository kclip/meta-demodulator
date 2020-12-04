from __future__ import print_function
import torch




def cross_entropy_loss(loss, error_rate, M, s, out):
    ## loss function
    K, _= s.size()
    success = 0
    loss = 0
    for i in range(K):
        if M == 2:
            if s[i, 0] == -1:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == 1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
        elif M == 4:
            if s[i, 0] == -1 and s[i, 1] == -1:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 1:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
        elif M == 5:
            if s[i, 0] == -3:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 3:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
        elif M == 16:
            if s[i, 0] == -3 and s[i, 1] == -3:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -3 and s[i, 1] == 1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -3:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1

            elif s[i, 0] == -3 and s[i, 1] == 3:
                loss = loss - out[i][4]
                if torch.argmax(out[i]) == 4:
                    success = success + 1
            elif s[i, 0] == 3 and s[i, 1] == 1:
                loss = loss - out[i][5]
                if torch.argmax(out[i]) == 5:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -1:
                loss = loss - out[i][6]
                if torch.argmax(out[i]) == 6:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == -3:
                loss = loss - out[i][7]
                if torch.argmax(out[i]) == 7:
                    success = success + 1

            elif s[i, 0] == 3 and s[i, 1] == 3:
                loss = loss - out[i][8]
                if torch.argmax(out[i]) == 8:
                    success = success + 1
            elif s[i, 0] == 3 and s[i, 1] == -1:
                loss = loss - out[i][9]
                if torch.argmax(out[i]) == 9:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == -1:
                loss = loss - out[i][10]
                if torch.argmax(out[i]) == 10:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 3:
                loss = loss - out[i][11]
                if torch.argmax(out[i]) == 11:
                    success = success + 1

            elif s[i, 0] == 3 and s[i, 1] == -3:
                loss = loss - out[i][12]
                if torch.argmax(out[i]) == 12:
                    success = success + 1
            elif s[i, 0] == -3 and s[i, 1] == -1:
                loss = loss - out[i][13]
                if torch.argmax(out[i]) == 13:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 1:
                loss = loss - out[i][14]
                if torch.argmax(out[i]) == 14:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 3:
                loss = loss - out[i][15]
                if torch.argmax(out[i]) == 15:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
    loss = loss/K #average over number of samples
    error_rate = 1 - success/K
    return [loss, error_rate]


def cross_entropy_loss_test(uncertainty_loss, loss, error_rate, M, s, out):
    ## loss function
    K, _= s.size()
    success = 0
    loss = 0
    uncertainty_loss = 0
    for i in range(K):
        if M == 2:
            uncertainty_loss = uncertainty_loss - torch.max(out[i])
            if s[i, 0] == -1:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == 1:
                loss = loss - out[i][1] # using logsoftmax
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
        elif M == 4:
            uncertainty_loss = uncertainty_loss - torch.max(out[i])
            if s[i, 0] == -1 and s[i, 1] == -1:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 1:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')
        elif M == 5:
            uncertainty_loss = uncertainty_loss - torch.max(out[i])
            if s[i, 0] == -3:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 3:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')

        elif M == 16:
            ### uncertainty loss ###
            uncertainty_loss = uncertainty_loss - torch.max(out[i])

            ########################
            if s[i, 0] == -3 and s[i, 1] == -3:
                loss = loss - out[i][0]
                if torch.argmax(out[i]) == 0:
                    success = success + 1
            elif s[i, 0] == -3 and s[i, 1] == 1:
                loss = loss - out[i][1]
                if torch.argmax(out[i]) == 1:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 1:
                loss = loss - out[i][2]
                if torch.argmax(out[i]) == 2:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -3:
                loss = loss - out[i][3]
                if torch.argmax(out[i]) == 3:
                    success = success + 1

            elif s[i, 0] == -3 and s[i, 1] == 3:
                loss = loss - out[i][4]
                if torch.argmax(out[i]) == 4:
                    success = success + 1
            elif s[i, 0] == 3 and s[i, 1] == 1:
                loss = loss - out[i][5]
                if torch.argmax(out[i]) == 5:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == -1:
                loss = loss - out[i][6]
                if torch.argmax(out[i]) == 6:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == -3:
                loss = loss - out[i][7]
                if torch.argmax(out[i]) == 7:
                    success = success + 1

            elif s[i, 0] == 3 and s[i, 1] == 3:
                loss = loss - out[i][8]
                if torch.argmax(out[i]) == 8:
                    success = success + 1
            elif s[i, 0] == 3 and s[i, 1] == -1:
                loss = loss - out[i][9]
                if torch.argmax(out[i]) == 9:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == -1:
                loss = loss - out[i][10]
                if torch.argmax(out[i]) == 10:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 3:
                loss = loss - out[i][11]
                if torch.argmax(out[i]) == 11:
                    success = success + 1

            elif s[i, 0] == 3 and s[i, 1] == -3:
                loss = loss - out[i][12]
                if torch.argmax(out[i]) == 12:
                    success = success + 1
            elif s[i, 0] == -3 and s[i, 1] == -1:
                loss = loss - out[i][13]
                if torch.argmax(out[i]) == 13:
                    success = success + 1
            elif s[i, 0] == -1 and s[i, 1] == 1:
                loss = loss - out[i][14]
                if torch.argmax(out[i]) == 14:
                    success = success + 1
            elif s[i, 0] == 1 and s[i, 1] == 3:
                loss = loss - out[i][15]
                if torch.argmax(out[i]) == 15:
                    success = success + 1
            else:
                NotImplementedError('something is wrong')

    loss = loss/K
    error_rate = 1 - success/K
    uncertainty_loss = uncertainty_loss/K
    return [uncertainty_loss, loss, error_rate]


