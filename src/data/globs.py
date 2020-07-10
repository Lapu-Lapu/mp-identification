TMP = {'params': ['numprim'],
       'scores': ['MSE', 'ELBO']}
DMP = {'params': ['npsi'],
       'scores': ['MSE']}
VCGPDM = dict(params=['dyn', 'lvm'],
              scores=['MSE', 'ELBO', 'dyn_elbo', 'lvm_elbo'])
MAPGPDM = {'params': [],
           'scores': ['MSE', 'ELBO']}
CATCH = {'params': [], 'scores': []}
model = {'vcgpdm': VCGPDM,
         'vgpdm': VCGPDM,
         'tmp': TMP,
         'dmp': DMP,
         'mapgpdm': MAPGPDM,
         'mapcgpdm': MAPGPDM,
         'catchtrial': CATCH}

pp = {'numprim': '# Primitives',
      'npsi': '# Basis functions',
      'vgpdm': 'vGPDM',
      'vcgpdm': 'vCGPDM',
      'MSE': 'MSE',
      'ELBO': 'ELBO (Total)',
      'dyn_elbo': 'ELBO (Dynamics)',
      'lvm_elbo': 'ELBO (Pose)',
      'tmp': 'TMP',
      'dmp': 'DMP',
      'mapcgpdm': 'cGPDM (MAP)',
      'mapgpdm': 'GPDM (MAP)',
      'map_cgpdm': 'cGPDM (MAP)',
      'map_gpdm': 'GPDM (MAP)'
      }

exp1_columns_raw = [
    'trial_number',
    'right',
    'flipped',
    'correct',
    'block',
    'left',
    'trials.thisTrialN',
    'trials.thisN',
    'participant_key_response',
    'date',
    'expName',
    'participant'
]

exp2_columns_raw = [
    'n_trial',
    'mov_file_generated',
    'mov_file_natural',
    'correct_answer',
    'response.corr',
    'mirrored',
    'switch_stim_pos',
    'condition',
    'participant',
    'age',
    'gender'
]
