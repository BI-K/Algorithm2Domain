def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class AbstractDatasetConfigClass():
        def __init__(self, 
                     shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None,
                     hidden_size_gru=32,
                     tcn_final_out_channels=32):


                self.shuffle = shuffle
                self.drop_last = drop_last
                self.normalize = normalize
                # Model configs
                self.kernel_size = kernel_size
                self.stride = stride
                self.dropout = dropout
                self.mid_channels = mid_channels
                # d_model
                self.final_out_channels = final_out_channels
                self.features_len = features_len
                # SWIFT features
                self.lstm_layers = lstm_layers
                self.bidirectional_lstm = bidirectional_lstm
                # TCN features
                self.tcn_layers = tcn_layers
                self.tcn_kernel_size = tcn_kernel_size
                self.ltcn_layers = ltcn_layers
                self.ode_unfolds = ode_unfolds
                self.cfcn_layers = cfcn_layers
                self.backbone_units = backbone_units
                self.backbone_layers = backbone_layers
                self.backbone_dropout = backbone_dropout
                self.lstm_hid = lstm_hid
                self.lstm_n_layers = lstm_n_layers
                self.lstm_bid = lstm_bid
                self.num_layers = num_layers
                self.pos_encoding = pos_encoding
                self.activation = activation
                self.norm = norm
                self.freeze = freeze
                self.n_heads = n_heads
                self.dim_feedforward = dim_feedforward
                self.disc_hid_dim = disc_hid_dim
                self.data_augmentation_configs = data_augmentation_configs
                self.hidden_size_gru = hidden_size_gru
                self.tcn_final_out_channels = tcn_final_out_channels


        def update_configs(self, params):
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        def to_dict(self):
            """Convert the configuration to a dictionary."""
            return {key: getattr(self, key) for key in self.__dict__ if not key.startswith('_')}

class HAR(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None):
        super(HAR, self).__init__()
        # dataset parameters
        self.scenarios = [("2", "11")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.num_cont_output_channels = 0
        self.sequence_len = 128
        self.num_classes = 6
        self.input_channels = 6
        
        
        
class EEG(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.num_cont_output_channels = 0
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
                          ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]
        self.input_channels = 6
        self.post_processing_classification = False
        

class WISDM(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
                          ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]
        self.num_classes = 6
        self.num_cont_output_channels = 0
        self.input_channels = 6
        self.post_processing_classification = False
        

class OHIO(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None
                 ):
        super(OHIO, self).__init__()
        self.class_names = ["Not Hypoglycemic", "Hypoglycemic"]
        self.scenarios = [("0", "1")]
        self.num_classes = 2
        self.input_channels = 1
        self.num_cont_output_channels = 0
        self.sequence_len = 128 # ???


class WEATHER(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None
                 ):
        super(WEATHER, self).__init__()

        # Temperature classes for discretization
        self.class_names = ["Below 25.5°C", "25.5-26.5°C", "26.5-27.5°C", "27.5-28.5°C",
                              "28.5-29.5°C", "29.5-30.5°C", "30.5-31.5°C", "Above 31.5°C"]
        # (1, 2) - Madrid with Bilbao
        # (0, 2) - Valencia with Bilbao
        # (4, 2) - Seville with Bilbao
        # (2, 1) - Bilbao with Madrid
        self.scenarios = [(1, 2), (0, 2), (4, 2), (2, 1)]
        self.num_classes = 8
        self.input_channels = 7
        self.num_cont_output_channels = 0
        self.sequence_len = 72
        self.post_processing_classification = False

        
class HHAR(AbstractDatasetConfigClass):  ## HHAR dataset, SAMSUNG device.
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None
                 ):
        super(HHAR, self).__init__()

        self.sequence_len = 128
        self.scenarios = [("0", "6")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.num_cont_output_channels = 0
        self.input_channels = 3
        self.post_processing_classification = False


        
        
class FD(AbstractDatasetConfigClass):
    def __init__(self,shuffle=True,
                     drop_last=True,
                     normalize=True, 
                     kernel_size=32,
                     stride=6,
                     dropout=0.1,
                     mid_channels=64,
                     final_out_channels=64,
                     features_len=1,
                     lstm_layers=[256, 16],
                     bidirectional_lstm=True,
                     tcn_layers=[75, 150],
                     tcn_kernel_size=17,
                     ltcn_layers=[64, 32],
                     ode_unfolds=2,
                     cfcn_layers=[64, 32],
                     backbone_units=32,
                     backbone_layers=1,
                     backbone_dropout=0.1,
                     lstm_hid=128,
                     lstm_n_layers=1,
                     lstm_bid=False,
                     num_layers = 1,
                     pos_encoding = 'fixed',
                     activation='gelu',
                     norm='BatchNorm',
                     freeze=False,
                     n_heads=8,
                     dim_feedforward=128,
                     disc_hid_dim=64,
                     data_augmentation_configs=None):
        super(FD, self).__init__()

        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"),("1", "3"),
                          ("2", "1"),("2", "3"),  ("3", "0"), ("3", "1"), ("3", "2")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.num_cont_output_channels = 0
        self.input_channels = 6
        self.post_processing_classification = False
        

# base config, if prediction of all six channels: cvp, hr, npb sys, nbp dias, nbp mean, spo2
#class PHD(AbstractDatasetConfigClass):
#    def __init__(self,shuffle=True, drop_last=True, kernel_size=32, dropout=0.1, mid_channels=64,
#                    final_out_channels=64, features_len=1, lstm_layers=[256, 16], bidirectional_lstm=True, tcn_layers=[150, 75],
#                     tcn_kernel_size=17, ltcn_layers=[64, 32], ode_unfolds=2, cfcn_layers=[64, 32], backbone_units=32,
#                     backbone_layers=1, backbone_dropout=0.1, lstm_hid=128, lstm_n_layers=1, lstm_bid=False, num_layers = 1,
#                     pos_encoding = 'fixed', activation='gelu', norm='BatchNorm', freeze=False, n_heads=8, dim_feedforward=128,
#                     disc_hid_dim=64,data_augmentation_configs=None):
#        super(PHD, self).__init__()
#        self.stride=1 # select stride = 1 to avoid padding with 
#        self.normalize = True # do not normalize as variance too little for cvp and hr for many batches (batch size 256)
#        self.sequence_len = 12
#        # self.scenarios = [("cardiac_surgery_dropped", "no_cardiac_surgery_dropped")]
#        self.scenarios = [("cardiac_surgery_cleaned", "no_cardiac_surgery_cleaned")] # TODO find a nice way to deal with these channels that have no variance - for normalization
#        self.class_names = []
#        self.num_classes = 0
#        self.num_cont_output_channels = 6
#        #self.input_channels = 4
#        self.input_channels = 6

# config continuous prediction, configure for spo2 or nbp_mean in scenarios
class PHD(AbstractDatasetConfigClass):
    def __init__(self,
                 dropout=0.5,
                 kernel_size=32,
                 mid_channels=64,
                 stride=6,
                 
                 activation='gelu',
                 backbone_dropout=0.1,
                 backbone_layers=1,
                 backbone_units=32,
                 bidirectional_lstm=True,
                 shuffle=True, drop_last=True,    
                    final_out_channels=64, features_len=1, lstm_layers=[256, 16],  tcn_layers=[75, 150],
                     tcn_kernel_size=17, ltcn_layers=[64, 32], ode_unfolds=2, cfcn_layers=[64, 32], 
                       lstm_hid=128, lstm_n_layers=1, lstm_bid=False, num_layers = 1,
                     pos_encoding = 'fixed',  norm='BatchNorm', freeze=False, n_heads=8, dim_feedforward=128,
                     disc_hid_dim=64,data_augmentation_configs=None, hidden_size_gru = 32, tcn_final_out_channels=32):
        super(PHD, self).__init__()
        self.normalize = True
        self.sequence_len = 12
        # self.scenarios = [("cardiac_surgery_cleaned_nbp_mean", "no_cardiac_surgery_cleaned_nbp_mean")]
        # self.scenarios = [("cardiac_surgery_cleaned_spo2", "no_cardiac_surgery_cleaned_spo2")]
        # self.scenarios = [("no_cardiac_surgery_cleaned_nbp_mean", "cardiac_surgery_cleaned_nbp_mean")]
        # self.scenarios = [("no_cardiac_surgery_cleaned_spo2", "cardiac_surgery_cleaned_spo2")]

        # self.scenarios = [("respiratory_surgery_cleaned_nbp_mean", "no_respiratory_surgery_cleaned_nbp_mean")]
        # self.scenarios = [("respiratory_surgery_cleaned_spo2", "no_respiratory_surgery_cleaned_spo2")]
        # self.scenarios = [("no_respiratory_surgery_cleaned_nbp_mean", "respiratory_surgery_cleaned_nbp_mean")]
        # self.scenarios = [("no_respiratory_surgery_cleaned_spo2", "respiratory_surgery_cleaned_spo2")]

        # self.scenarios = [("ventilation_cleaned_nbp_mean", "no_ventilation_cleaned_nbp_mean")]
        # self.scenarios = [("ventilation_cleaned_spo2", "no_ventilation_cleaned_spo2")]
        # self.scenarios = [("no_ventilation_cleaned_nbp_mean", "ventilation_cleaned_nbp_mean")]
        # self.scenarios = [("no_ventilation_cleaned_spo2", "ventilation_cleaned_spo2")]

        # self.scenarios = [("vasopressors_cleaned_nbp_mean", "no_vasopressors_cleaned_nbp_mean")]
        self.scenarios = [("vasopressors_cleaned_spo2", "no_vasopressors_cleaned_spo2")]
        # self.scenarios = [("no_vasopressors_cleaned_nbp_mean", "vasopressors_cleaned_nbp_mean")]
        # self.scenarios = [("no_vasopressors_cleaned_spo2", "vasopressors_cleaned_spo2")]

        self.class_names = ["0", "1"]
        self.num_classes = 2
        self.post_processing_classification = True 
        self.num_cont_output_channels = 1
        self.input_channels = 6        

#class PHD(AbstractDatasetConfigClass):
#    def __init__(self,shuffle=True, drop_last=True, kernel_size=32, stride=6, dropout=0.1, mid_channels=64,
#                    final_out_channels=64, features_len=1, lstm_layers=[256, 16], bidirectional_lstm=True, tcn_layers=[75, 150],
#                     tcn_kernel_size=17, ltcn_layers=[64, 32], ode_unfolds=2, cfcn_layers=[64, 32], backbone_units=32,
#                     backbone_layers=1, backbone_dropout=0.1, lstm_hid=128, lstm_n_layers=1, lstm_bid=False, num_layers = 1,
#                     pos_encoding = 'fixed', activation='gelu', norm='BatchNorm', freeze=False, n_heads=8, dim_feedforward=128,
#                     disc_hid_dim=64,data_augmentation_configs=None):
#        super(PHD, self).__init__()
#        self.normalize = False
#        self.sequence_len = 12
#        self.scenarios = [("cardiac_surgery_hypotension_cleaned", "no_cardiac_surgery_hypotension_cleaned")]

#        self.class_names = ["0", "1"]
#        self.num_classes = 2
#        self.num_cont_output_channels = 0
#        self.input_channels = 6  
#        self.post_processing_classification = False 