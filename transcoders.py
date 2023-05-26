import numpy as np
import yaml
import numpy as np
import numpy as np
import torch.utils.data
import torch
import utils.pinv_transcoder as pt
import models as md
import yaml
from pann.pann_mel_inference import PannMelInference
from yamnet.yamnet_mel_inference import YamnetMelInference
from utils.util import get_transforms

class ThirdOctaveToMelTranscoderPinv():

    def __init__(self, model_path, model_name, device, classifier='PANN'):

        self.device = device

        #MODELS_PATH = project_data_path / 'exp_models' / ('exp_models_' + transcoder)
        # MODELS_PATH = project_data_path / 'model'

        #SETTINGS_PATH = project_data_path / 'exp_settings' 
        # SETTINGS_PATH = project_data_path.parent.parent / "data_settings"
        #settings = ut.load_settings(Path(SETTINGS_PATH / (config+'_settings'+'.yaml')))

        with open(model_path + "/" + model_name + '_settings.yaml') as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        self.input_shape = settings_model.get('input_shape')
        self.output_shape = settings_model.get('output_shape')

        cnn_kernel_size = settings_model.get('cnn_kernel_size')
        cnn_dilation = settings_model.get('cnn_dilation')
        cnn_nb_layers = settings_model.get('cnn_nb_layers')
        cnn_nb_channels = settings_model.get('cnn_nb_channels')

        mlp_hl_1 = settings_model.get('mlp_hl_1')
        mlp_hl_2 = settings_model.get('mlp_hl_2')

        #from settings of the model
        classifier = settings_model.get('mels_type')

        self.tho_tr, self.mels_tr = get_transforms(sr=32000, 
                                            flen=4096,
                                            hlen=4000,
                                            classifier=classifier,
                                            device=device)

        # self.tho_tr = tho_tr
        # self.mels_tr = mels_tr

    def transcode_from_thirdo(self, x, dtype=torch.FloatTensor):
        x_inf = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], device=self.device)
        x_inf = x_inf[0].T.numpy()
        return(x_inf)
    
    def transcode_from_1s_wav(self, x, dtype=torch.FloatTensor):
        x_tho = self.tho_tr.wave_to_third_octave(x)
        x_tho = torch.from_numpy(x_tho.T)
        x_tho = x_tho.unsqueeze(0)
        x_tho = x_tho.type(dtype)
        x_mels_inf = self.transcode_from_thirdo(x_tho)
        return(x_mels_inf)
    
    def transcode_from_wav(self, x):
        chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        for x_i in x_sliced:
            x_mels_inf = self.transcode_from_1s_wav(x_i)
            x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
        return(x_mels_inf_tot)

class ThirdOctaveToMelTranscoder():

    def __init__(self, transcoder, model_name, model_path, device):

        self.device = device

        #MODELS_PATH = project_data_path / 'exp_models' / ('exp_models_' + transcoder)
        #model_path = project_data_path / 'model'

        #SETTINGS_PATH = project_data_path / 'exp_settings' 
        # SETTINGS_PATH = project_data_path.parent.parent / "data_settings"

        #settings = ut.load_settings(Path(SETTINGS_PATH / (config+'_settings'+'.yaml')))

        with open(model_path + "/" + model_name + '_settings.yaml') as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        input_shape = settings_model.get('input_shape')
        output_shape = settings_model.get('output_shape')

        cnn_kernel_size = settings_model.get('cnn_kernel_size')
        cnn_dilation = settings_model.get('cnn_dilation')
        cnn_nb_layers = settings_model.get('cnn_nb_layers')
        cnn_nb_channels = settings_model.get('cnn_nb_channels')

        mlp_hl_1 = settings_model.get('mlp_hl_1')
        mlp_hl_2 = settings_model.get('mlp_hl_2')

        #from settings of the model
        classifier = settings_model.get('mels_type')

        print('CLASSIFIER')
        print(classifier)
        self.tho_tr, self.mels_tr = get_transforms(sr=32000, 
                                                flen=4096,
                                                hlen=4000,
                                                classifier=classifier,
                                                device=device)

        
        if transcoder == "cnn_pinv":
            self.model = md.CNN(input_shape=input_shape, output_shape=output_shape, tho_tr=self.tho_tr, mels_tr=self.mels_tr, kernel_size=cnn_kernel_size, dilation=cnn_dilation, nb_layers=cnn_nb_layers, nb_channels=cnn_nb_channels, device=device)
        if transcoder == "mlp":
            self.model = md.MLP(input_shape, output_shape, hl_1=mlp_hl_1, hl_2=mlp_hl_2)

        state_dict = torch.load(model_path + "/" + model_name + ".pth", map_location=device)
        self.model.load_state_dict(state_dict)

        # self.tho_tr = tho_tr
        # self.mels_tr = mels_tr

        if classifier == 'PANN':
            self.classif_inference = PannMelInference(verbose=False)
        if classifier == 'YamNet':
            self.classif_inference = YamnetMelInference(verbose=False)

    def transcode_from_thirdo(self, x):
        x_inf = self.model(x).detach()
        x_inf = x_inf[0].T.numpy()
        return(x_inf)
    
    def transcode_from_1s_wav(self, x, dtype=torch.FloatTensor):
        x_tho = self.tho_tr.wave_to_third_octave(x)
        x_tho = torch.from_numpy(x_tho.T)
        x_tho = x_tho.unsqueeze(0)
        x_tho = x_tho.type(dtype)
        x_mels_inf = self.transcode_from_thirdo(x_tho)
        return(x_mels_inf)
    
    def transcode_from_wav(self, x):
        chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = []
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
        for x_i in x_sliced:
            x_mels_inf = self.transcode_from_1s_wav(x_i)
            x_logits = self.mels_to_logit(x_mels_inf)
            x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
            x_logits_tot =np.concatenate((x_logits_tot, x_logits), axis=1)
            #x_mels_inf_tot.append(x_mels_inf)
        x_mels_inf_tot = np.array(x_mels_inf_tot)
        x_logits_tot = np.array(x_logits_tot)
        # x_mels_inf_tot = np.expand_dims(x_mels_inf_tot, axis=2)
        # x_mels_inf_tot = np.transpose(x_mels_inf_tot, (1, 3, 2, 0))
        #x_logits_tot = self.mels_to_logit_sliced(x_mels_inf_tot)
        #x_logits_tot = self.mels_to_logit(x_mels_inf_tot)
        return(x_mels_inf_tot, x_logits_tot)

    def transcode_from_wav_entire_file(self, x):
        chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
        for k, x_i in enumerate(x_sliced):
            x_mels_inf = self.transcode_from_1s_wav(x_i)
            if k == len(x_sliced)-1:   
                x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
            else:
                x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf[:, :-1]), axis=1)
            
        x_mels_inf_tot = np.array(x_mels_inf_tot)
        return(x_mels_inf_tot)

    def wave_to_mels_sliced(self, x):
        chunk_size = self.mels_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        X = []
        for x_i in x_sliced:
            xi_mels = self.mels_tr.wave_to_mels(x_i)
            X.append(xi_mels)
        X = np.array(X)
        X = np.expand_dims(X, axis=2)
        X = np.transpose(X, (1, 3, 2, 0))
        return(X)
    
    def mels_to_logit(self, x):
        temp = torch.from_numpy(np.expand_dims(np.expand_dims(x.T, axis=0), axis=0))
        temp =  temp.to(torch.float32)
        x_logits = self.classif_inference.simple_inference(temp, no_grad=True, mean=True)
        #COMMENT EVERYTHING IF MEAN IS TAKEN
        if len(x_logits.shape) > 2: 
            # Determine the number of full batches of 101 samples that can be extracted from the tensor
            num_batches = x_logits.size(1) // 101

            # Slice the tensor to remove any extra samples that don't fit into a full batch
            x_logits = x_logits[:, :num_batches * 101, :]

            # Reshape the tensor to split the second dimension into batches of 101 samples
            x_logits = x_logits.reshape(1, -1, 101, 527)
            #print('BEFORE')
            #print(x_logits.shape)
            # Compute the maximum value along the second dimension (i.e., the batch dimension)
            x_logits, _ = torch.max(x_logits, dim=2)
            #print('AFTER')
            #print(x_logits.shape)

            # The resulting tensor will have size [1, 10, 527], where 10 is the number of batches of 101 samples in the original tensor
            #print(x_logits.size())

        x_logits = x_logits.numpy().T
        return(x_logits)

    def mels_to_logit_entire_file(self, x, slice=True, batch_size=100):
        temp = torch.from_numpy(np.expand_dims(np.expand_dims(x.T, axis=0), axis=0))
        temp =  temp.to(torch.float32)
        x_logits = self.classif_inference.simple_inference(temp, no_grad=True, mean=not slice)

        if (len(x_logits.shape) > 2 & slice==True): 
            x_logits = x_logits[:, :-1, :]
            # Determine the number of full batches of 101 samples that can be extracted from the tensor
            num_batches = x_logits.size(1) // batch_size

            # Slice the tensor to remove any extra samples that don't fit into a full batch
            x_logits = x_logits[:, :num_batches * batch_size, :]

            # Reshape the tensor to split the second dimension into batches of 101 samples
            x_logits = x_logits.reshape(1, -1, batch_size, x_logits.shape[2])

            # Compute the maximum (or mean) value along the second dimension (i.e., the batch dimension)
            x_logits = torch.mean(x_logits, dim=2)
            x_logits, _ = torch.max(x_logits, dim=1)
            #x_logits = torch.mean(x_logits, dim=1)

        x_logits = x_logits.numpy().T
        return(x_logits)
    
    def mels_to_logit_sliced(self, x):
        X = []
        if len(x.shape) > 2:
            for i in range(x.shape[-1]):
                xi = x[:, :, 0, i]
                temp = torch.from_numpy(np.expand_dims(np.expand_dims(xi.T, axis=0), axis=0))
                temp =  temp.to(torch.float32)
                x_logits = self.classif_inference.simple_inference(temp, no_grad=True)
                x_logits = x_logits.numpy().T
                X.append(x_logits)
            X = np.array(X)
            #X = X.max(axis=0)
            X = X.mean(axis=0)
        else:
            temp = torch.from_numpy(np.expand_dims(np.expand_dims(x.T, axis=0), axis=0))
            temp =  temp.to(torch.float32)
            x_logits = self.classif_inference.simple_inference(temp, no_grad=True)
            X = x_logits.numpy().T
        return(X)