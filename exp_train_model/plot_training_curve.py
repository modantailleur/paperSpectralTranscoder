import numpy as np
import matplotlib.pyplot as plt
import main_doce_training as main_doce_training

"""
This file allows the plot of training curve. Please uncomment the row of
the model you want to plot and modify the parameters as you want.
Note that the plan must be specified explicitely (hybridts, cnn or ts).
"""

#hybrid ts
plan="hybridts"
selector = {"step":"compute", "classifier":"PANN", "dataset":"full", "prop_logit":100, "epoch":100, "dilation":1, "learning_rate":-3}

#non-ts
#plan="cnn"
#selector = {"transcoder":"cnn_pinv", "step":"train", "classifier":"PANN", "dataset":"full", "learning_rate":-3, "kernel_size":5, "nb_layers":5, "dilation":1, "nb_channels":64}

#ts
#plan="ts"
#selector = {"step":"compute", "classifier":"PANN", "dataset":"full", "transcoder":"effnet_b7", "epoch":150, "learning_rate":-5}
#selector = {"step":"compute", "classifier":"PANN", "dataset":"full", "transcoder":"effnet_b0", "epoch":100, "learning_rate":-5}
#selector = {"step":"compute", "classifier":"PANN", "dataset":"full", "transcoder":"self", "epoch":100, "learning_rate":-4}
#selector = {"step":"compute", "classifier":"PANN", "dataset":"full", "transcoder":"self", "epoch":50, "learning_rate":-4}

(data_train_total, settings_train_total, header_train_total) = main_doce_training.experiment.get_output(
  output = 'losses_train',
  selector = selector,
  path = "loss",
  plan = plan
  )

(data_valid_total, settings_valid_total, header_train_total) = main_doce_training.experiment.get_output(
  output = 'losses_valid',
  selector = selector,
  path = "loss",
  plan = plan
  )

(data_eval_total, settings_eval_total, header_eval_total) = main_doce_training.experiment.get_output(
  output = 'losses_eval',
  selector = selector,
  path = "loss",
  plan = plan
  )

for idx, (data_train, data_valid, data_eval,
          settings_train, settings_valid,
          settings_eval) in enumerate(zip(data_train_total, data_valid_total, data_eval_total,
                                          settings_train_total, settings_valid_total, settings_eval_total)):
    cmap = plt.cm.get_cmap('Set2')

    settings_train = settings_train.replace(' ', '+')  
    settings_train = settings_train.replace(',', '')  

    X_train = list(range(0, len(data_train)))
    X_valid = list(range(0, len(data_valid)))
    X_eval = list(range(0, len(data_eval)))
    
    X_train=np.array(X_train)
    X_valid = np.array(X_valid)
    X_eval = np.array(X_eval)
    
    X_valid = X_valid*int(len(X_train)/len(X_valid))
    X_eval = X_eval*int(len(X_train)/len(X_eval))
    
    plt.plot(X_train, data_train, label='train subset', c=cmap(2))
    plt.plot(X_valid, data_valid, '+-', label='validation subset', c=cmap(6))

    plt.ylim(0, 0.04)

    plt.xlabel('Iterations')
    if plan in ["ts", "hybridts"]:
      plt.ylabel('BCE loss')
    else:
      plt.ylabel('Mean Squared Error')

    plt.legend()
    
    plt.savefig('./figures_training_curves/' + str(settings_train) + '_training_curve.png', bbox_inches='tight')
    plt.clf()
