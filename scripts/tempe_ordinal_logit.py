# tempe_ordinal_logit.py

import pandas as pd
import biogeme
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.messaging as msg
from biogeme import models
from biogeme.expressions import Beta, log, Elem
from biogeme.distributions import logisticcdf

# Read the data
df = pd.read_csv('tempe_cleaneddata.csv', sep='\t')
df = df.sample(frac=0.8, random_state=100)
database = db.Database('tempe', df)
db_var = database.variables
globals().update(db_var)

# Parameters
ASC_noinjury = Beta('ASC_noinjury', 0, None, None, 1)
ASC_possinjury = Beta('ASC_possinjury', 0, None, None, 0)
ASC_nonincap = Beta('ASC_nonincap', 0, None, None, 0)
ASC_incap = Beta('ASC_incap', 0, None, None, 0)
ASC_fatal = Beta('ASC_fatal', 0, None, None, 0)

variables = [
    ['age', (1, 0, 0, 0, 0)],
    ['total_injuries', (1, 0, 0, 0, 0)],
    ['alcohol', (1, 0, 0, 0, 0)],
#     ['hour_morning', (1, 0, 0, 0, 0)], 
    ['hour_afternoon', (1, 0, 0, 0, 0)], 
    ['hour_night', (1, 0, 0, 0, 0)], 
#     ['hour_latenight', (1, 0, 0, 0, 0)], 
    ['nonintersection', (1, 0, 0, 0, 0)], 
#     ['light_daylight', (1, 0, 0, 0, 0)], 
    ['light_darklighted', (1, 0, 0, 0, 0)], 
    ['light_darknotlighted', (1, 0, 0, 0, 0)], 
#     ['light_dawndusk', (1, 0, 0, 0, 0)], 
    ['meteo_cloudy', (1, 0, 0, 0, 0)], 
    ['meteo_rain', (1, 0, 0, 0, 0)], 
    ['surf_wet', (1, 0, 0, 0, 0)], 
    ['type_cyclist', (1, 0, 0, 0, 0)], 
    ['type_pedestrian', (1, 0, 0, 0, 0)], 
    ['type_driverless', (1, 0, 0, 0, 0)], 
    ['action_turn', (1, 0, 0, 0, 0)], 
    ['action_lanes', (1, 0, 0, 0, 0)], 
    ['action_straight', (1, 0, 0, 0, 0)], 
    ['action_slowing', (1, 0, 0, 0, 0)], 
    ['cause_speeding', (1, 0, 0, 0, 0)], 
    ['cause_yield', (1, 0, 0, 0, 0)], 
    ['cause_unsafe', (1, 0, 0, 0, 0)], 
    ['cause_signal', (1, 0, 0, 0, 0)], 
    ['cause_following', (1, 0, 0, 0, 0)], 
    ['cause_distraction', (1, 0, 0, 0, 0)],
    ['cause_turn', (1, 0, 0, 0, 0)], 
    ['acc_rearend', (1, 0, 0, 0, 0)], 
    ['acc_leftturn', (1, 0, 0, 0, 0)], 
    ['acc_sides', (1, 0, 0, 0, 0)], 
#     ['acc_angle', (1, 0, 0, 0, 0)],
]

B = {k: [Beta(k+'_'+str(n), 0, None, None, j) for n,j in enumerate(a)] for (k, a) in variables}

# Parameters for the ordered logit. 
# tau1 <= 0
tau1 = Beta('tau1', -1, None, 0, 0)
# delta2 >= 0
delta2 = Beta('delta2', 1, 0, None, 0)
tau2 = tau1 + delta2
delta3 = Beta('delta3', 1, 0, None, 0)
tau3 = tau2 + delta3
delta4 = Beta('delta4', 1, 0, None, 0)
tau4 = tau3 + delta4

U = sum([B[n][1]*db_var[n] for n, _ in variables])

ChoiceProba = {
    0: 1 - logisticcdf(U-tau1),
    1: logisticcdf(U-tau1) - logisticcdf(U-tau2),
    2: logisticcdf(U-tau2) - logisticcdf(U-tau3),
    3: logisticcdf(U-tau3) - logisticcdf(U-tau4),
    4: logisticcdf(U-tau4)
}

av = {0:1, 1:1, 2:1, 3:1, 4:1}

logprob = log(Elem(ChoiceProba, severity))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
logger.setGeneral()
# logger.setDebug()

biogeme = bio.BIOGEME(database, logprob, userNotes="Ordinal logit, 80%data, full")
biogeme.modelName = 'm1_ordinallogit'
biogeme.calculateNullLoglikelihood(av)

results = biogeme.estimate()