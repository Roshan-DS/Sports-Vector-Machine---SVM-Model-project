import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

# 1
print(aaron_judge.columns)

# 2
print(aaron_judge.description.unique())

# 3
print(aaron_judge.type.unique())

# 4
def find_strike_zone(data_set):
  data_set['type'] = data_set['type'].map({'S':1, 'B':2})

  # 5
  print(data_set.type)

  # 6
  print(data_set['plate_x'])

  # 7
  data_set = data_set.dropna(subset = ['plate_x', 'plate_z', 'type'])

  # 8

  fig, ax = plt.subplots()

  plt.scatter(x = data_set['plate_x'], y = data_set['plate_z'], c = data_set['type'],cmap = plt.cm.coolwarm, alpha = 0.25)


  # 9
  training_set, validation_set = train_test_split(data_set, random_state = 1)

  # 10
  largest = {'value': 0, 'gamma': 1, 'C': 1}
  for gamma in range(1,5):
    for C in range(1,5):
      classifier = SVC(kernel = 'rbf', gamma = gamma, C = C)
      classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
      score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']])
      if (score > largest['value']):
        largest['value'] = score
        largest['gamma'] = gamma
        largest['C'] = C

  print(largest)


  # 16
  ax.set_ylim(-2,6)
  ax.set_xlim(-3,3)
  draw_boundary(ax, classifier)
  plt.show()

print(find_strike_zone(aaron_judge))

