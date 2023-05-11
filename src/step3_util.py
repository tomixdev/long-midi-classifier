import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

import params


def set_plt_config():
    # デフォルトフォントサイズ変更
    plt.rcParams['font.size'] = 14
    # デフォルトグラフサイズ変更
    plt.rcParams['figure.figsize'] = (6,6)
    # デフォルトで方眼表示ON
    plt.rcParams['axes.grid'] = True
    

def set_np_config():
    # numpyの表示桁数設定
    np.set_printoptions(suppress=True, precision=5)
    

def set_and_get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print (f"{device = }")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print (f"{device = }")
    else:
        device = torch.device('cpu')
        print ("No GPU device found. Running on CPU!")

    return device


def torch_seed (seed=params.RAND_SEED):
    # PyTorch乱数固定 
    
    # CPUに対して、乱数の種
    torch.manual_seed(seed)

    if torch.has_mps:
        try:
            torch.mps.manual_seed(seed)
        except:
            raise Exception("""
                torch.mps.manual_seed is AVAILABLE from pytorch 2.0. This is NOT AVAILABLE for previous versions of pytorch. torch.mps.manual_seed MUST be fixed to ensure reproducibility. To ensure random seed is fixed, run the following code. If the result is always the same, then it should be okay:

                import torch

                torch.manual_seed(42)
                torch.mps.manual_seed(42)

                x = torch.randn(3, 3, device="mps")
                print(x)
            """)
        torch.use_deterministic_algorithms = True
    elif torch.cuda.is_available():
        # GPUにたいして、乱数の種。
        torch.cuda.manual_seed(seed)
        # GPUを利用する場合、Performanceの最適化を図るため、値がぴったりいくつになると保証されない場合がある。
        # この問題にたいおうするための設定がdeterministic・
        # この項目をTrueに設定すると、Performanceより値の再現性重視で、GPUが同じ結果を返すようになる。
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.deterministic = True
    else:
        pass
    
    
# 学習関数
# historyのargumentがあることで、追加学習に対応できるようにする。
def fit (net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    '''
    :param net:
    :param optimizer:
    :param criterion:
    :param num_epochs:
    :param train_loader:
    :param test_loader:
    :param device:
    :param history:
    :return: history (epoch_count+1, avg_train_loss, avg_validation_acc, avg_validation_loss, avg_validation_acc)
    '''

    # tqdm ライブラリのimport
    from tqdm.notebook import tqdm

    initial_epoch_count = len (history)

    for epoch_count in range (initial_epoch_count, initial_epoch_count+num_epochs):
        train_loss_value = 0
        train_acc_value = 0
        validation_loss_value = 0
        validation_acc_value = 0

        # 訓練フェーズ ---------------------------------------------------------------------------
        net.train() 

        train_data_count = 0

        for train_inputs, train_ground_truth_labels in tqdm(train_loader):
            train_data_count += len (train_ground_truth_labels)
            train_inputs = train_inputs.to(device)
            train_ground_truth_labels = train_ground_truth_labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            train_outputs = net (train_inputs)

            # 損失計算
            train_loss = criterion(train_outputs, train_ground_truth_labels)
            train_loss_value += train_loss.item()

            # 勾配計算
            train_loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測値算出
            predicted_labels_train = torch.max(train_outputs, 1)[1]

            # 正解件数算出
            train_acc_value += (predicted_labels_train == train_ground_truth_labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss_value / train_data_count
            avg_train_acc = train_acc_value / train_data_count

        # 予測フェーズ　---------------------------------------------------------------------
        net.eval() # このfunctionは、モデルクラスを定義するときに利用している親クラスnn.Moduleで定義されている。
                    # nn.Dropoutや、nn.BatchNorm2d といったレイヤー関数では、それぞれの関数に対して「今は訓練フェーズ」「今は予測フェーズ」
                    # ということを、教える必要があるので、ここに入っている。
        test_data_count = 0

        for test_inputs, test_ground_truth_labels in test_loader:
            test_data_count += len (test_ground_truth_labels)
            test_inputs = test_inputs.to(device)
            test_ground_truth_labels = test_ground_truth_labels.to(device)

            # 予測計算
            test_outputs = net(test_inputs)

            # 損失計算
            test_loss = criterion(test_outputs, test_ground_truth_labels)
            validation_loss_value += test_loss.item()

            # 予測値算出
            predicted_labels_test = torch.max(test_outputs, 1)[1]

            # print the device information of predicted_labels_test

            # 正解件数算出
            validation_acc_value += (predicted_labels_test == test_ground_truth_labels).sum().item()

            # 損失と精度の計算
            avg_validation_loss = validation_loss_value / test_data_count
            avg_validation_acc = validation_acc_value / test_data_count

        item = np.array([epoch_count+1, avg_train_loss, avg_train_acc, avg_validation_loss, avg_validation_acc])
        history = np.vstack((history, item))

    return history


def evaluate_history (history):
    if isinstance (history, pd.DataFrame):
        # convert pd.DataFrame to numpy.array
        history = history.values
    
    assert isinstance(history, np.ndarray)

    # 損失と精度の確認
    print ("-------for train data ------------")
    print ("initial loss = ")
    print (history [0, 1])
    print ("initial accuracy = ")
    print (history[0, 2])
    print ("final loss = ")
    print (history [-1, 1])
    print ("final accuracy = ")
    print (history [-1, 2])

    print ()

    print ("-------for test data -----------")
    print ("initial loss = ")
    print (history [0, 3])
    print ("initial accuracy = ")
    print (history[0, 4])
    print ("final loss = ")
    print (history [-1, 3])
    print ("final accuracy = ")
    print (history [-1, 4])

    num_epochs = len (history)
    unit = num_epochs / 10

    # 学習曲線の表示　(loss について)
    plt.figure(figsize = (9,8))
    plt.plot(history[:, 0], history[:, 1], 'b', label='train')
    plt.plot(history[:, 0], history[:, 3], 'k', label='test')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning curve for loss')
    plt.legend()
    plt.show()

    # 学習曲線の表示 (accuracyについて)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history [:, 2], 'b', label="train")
    plt.plot(history[:, 0], history [:, 4], 'k', label = "test")
    plt.xticks(np.arange(0, num_epochs, unit))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title ('learning curve for accuracy')
    plt.legend()
    plt.show()
