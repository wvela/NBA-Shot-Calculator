import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import playercareerstats

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle, Arc

# Starting 5 players from each NBA team
AtlantaHawks = {"Atlanta Hawks": ["Trae Young", "Kevin Huerter", "DeAndre' Bembry", "John Collins", "Alex Len"]}
BostonCeltics = {"Boston Celtics": ["Kemba Walker", "Jaylen Brown", "Gordon Hayward", "Jayson Tatum", "Enes Kanter"]}
BrooklynNets = {"Brooklyn Nets": ["Kyrie Irving", "Joe Harris", "Caris LeVert", "Rodions Kurucs", "DeAndre Jordan"]}
CharlotteHornets = {"Charlotte Hornets": ["Terry Rozier", "Malik Monk", "Nicolas Batum", "Marvin Williams", "Cody Zeller"]}
ChicagoBulls = {"Chicago Bulls": ["Coby White", "Zach LaVine", "Kris Dunn", "Lauri Markkanen", "Wendell Carter Jr."]}
ClevelandCavaliers = {"Cleveland Cavaliers": ["Collin Sexton", "Darius Garland", "Cedi Osman", "Kevin Love", "Tristan Thompson"]}
DallasMavericks = {"Dallas Mavericks": ["Delon Wright", "Luka Doncic", "Tim Hardaway Jr.", "Kristaps Porzingis", "Dwight Powell"]}
DenverNuggets = {"Denver Nuggets": ["Jamal Murray", "Gary Harris", "Torrey Craig", "Paul Millsap", "Nikola Jokic"]}
DetroitPistons = {"Detroit Pistons": ["Reggie Jackson", "Luke Kennard", "Tony Snell", "Blake Griffin", "Andre Drummond"]}
GoldenStateWarriors = {"Golden State Warriors": ["Stephen Curry", "D'Angelo Russell", "Alfonzo McKinnie", "Draymond Green", "Willie Cauley-Stein"]}
HoustonRockets = {"Houston Rockets": ["Russell Westbrook", "James Harden", "Eric Gordon", "P.J. Tucker", "Clint Capela"]}
IndianaPacers = {"Indiana Pacers": ["Malcolm Brogdon", "Jeremy Lamb", "T.J. Warren", "Domantas Sabonis", "Myles Turner"]}
LosAngelesLakers = {"Los Angeles Lakers": ["Rajon Rondo", "Danny Green", "LeBron James", "Anthony Davis", "JaVale McGee"]}
MemphisGrizzlies = {"Memphis Grizzlies": ["Ja Morant", "Dillon Brooks", "Jae Crowder", "Jaren Jackson Jr.", "Jonas Valanciunas"]}
MiamiHeat = {"Miami Heat": ["Goran Dragic", "Jimmy Butler", "Justise Winslow", "James Johnson", "Bam Adebayo"]}
MilwaukeeBucks = {"Milwaukee Bucks": ["Eric Bledsoe", "Wesley Matthews", "Khris Middleton", "Giannis Antetokounmpo", "Brook Lopez"]}
MinnesotaTimberwolves = {"Minnesota Timberwolvers": ["Jeff Teague", "Josh Okogie", "Andrew Wiggins", "Robert Covington", "Karl-Anthony Towns"]}
NewOrleansPelicans = {"New Orleans Pelicans": ["Lonzo Ball", "Jrue Holiday", "Brandon Ingram", "Zion Williamson", "Derrick Favors"]}
NewYorkKnicks = {"New York Knicks": ["Dennis Smith Jr.", "RJ Barrett", "Reggie Bullock", "Julius Randle", "Mitchell Robinson"]}
OklahomaCityThunder = {"Ohlahoma City Thunder": ["Chris Paul", "Shai Gilgeous-Alexander", "Terrance Ferguson", "Danilo Gallinari", "Steven Adams"]}
OrlandoMagic = {"Orlando Magic": ["D.J. Augustin", "Evan Fournier", "Jonathan Isaac", "Aaron Gordon", "Nikola Vucevic"]}
PhiladelphiaSixers = {"Philadelphia Sixers": ["Ben Simmons", "Josh Richardson", "Tobias Harris", "Al Horford", "Joel Embiid"]}
PhoenixSuns = {"Phoenix Suns": ["Ricky Rubio", "Devin Booker", "Mikal Bridges", "Dario Saric", "Deandre Ayton"]}
PortlandTrailblazers = {"Portland Trailblazers": ["Damian Lillard", "CJ McCollum", "Rodney Hood", "Zach Collins", "Hassan Whiteside"]}
SacramentoKings = {"Sacramento Kings": ["De'Aaron Fox", "Buddy Hield", "Harrison Barnes", "Marvin Bagley III", "Dewayne Dedmon"]}
SanAntonioSpurs = {"San Antonio Spurs": ["Derrick White", "Bryn Forbes", "DeMar DeRozan", "Rudy Gay", "LaMarcus Aldridge"]}
TorontoRaptors = {"Toronto Raptors": ["Kyle Lowry", "Norman Powell", "OG Anunoby", "Pascal Siakam", "Marc Gasol"]}
UtahJazz = {"Utah Jazz": ["Mike Conley", "Donovan Mitchell", "Bojan Bogdanovic", "Joe Ingles", "Rudy Gobert"]}
WashingtonWizards = {"Washington Wizards": ["Ish Smith", "Bradley Beal", "Troy Brown Jr.", "Rui Hachimura", "Thomas Bryant"]}

# List of all NBA Teams
NBAList = [AtlantaHawks, BostonCeltics, BrooklynNets, CharlotteHornets, ChicagoBulls, ClevelandCavaliers, DallasMavericks, DenverNuggets, DetroitPistons, GoldenStateWarriors, 
          HoustonRockets, IndianaPacers, LosAngelesLakers, MemphisGrizzlies, MiamiHeat, MilwaukeeBucks, MinnesotaTimberwolves, NewOrleansPelicans, NewYorkKnicks, OklahomaCityThunder,
          OrlandoMagic, PhiladelphiaSixers, PhoenixSuns, PortlandTrailblazers, SacramentoKings, SanAntonioSpurs, TorontoRaptors, UtahJazz, WashingtonWizards]

smallNBAList = [ChicagoBulls, LosAngelesLakers, WashingtonWizards]

# Returns a DF of career stats for each player on a given input team
"""
Pre-Conditions:
                If looking for a single teams stats:
                    Parameters = teamName   = "Name of teame" ex. "Washington Wizards"
                                 playerName = Team name dictionary ex. WashingtonWizards

                If looking for stats for all NBA teams:
                    Parameters = NBAList = List of all NBA teams ex. NBAList (Global Variable)
                    * Can be used to find any number of NBA team stats. List must be in format:
                                [{"Team Name": [List of five players]}, {"Team Name": [List of five players]}, ... ]
"""
def getPlayerShotChartDetail(seasonID = "2019-20", NBAList = None, teamName = None, playerName = None):
  nbaPlayers = players.get_players()
  NBATeams = [] 
  if NBAList is None:
    # Create Dict of starting 5 players for a given team 
    teamDict = {}
    teamList = []
    teamName = list(playerName.keys())[0]
    i = 0
    while (i < 5):
      playerDict = [player for player in nbaPlayers if player['full_name'] == playerName[teamName][i]][0]
      teamList.append(playerDict)
      #print(playerDict)
      i += 1
    teamDict[teamName] = teamList
    NBATeams.append(teamDict)
  else:
    # Create Dict of starting 5 players for all teams in NBAList
    for team in NBAList:
      teamDict = {}
      teamList = []
      teamName = list(team.keys())[0]
      i = 0
      while (i < 5):
        playerDict = [player for player in nbaPlayers if player['full_name'] == team[teamName][i]][0]
        teamList.append(playerDict)
        i += 1
      teamDict[teamName] = teamList
      NBATeams.append(teamDict)

  #print(NBATeams)

  # Create a DF for each player on the team
  statDFs = {}
  for team in NBATeams:
    i = 0
    statDFList = []
    teamName = list(team.keys())[0]
    while (i < 5):
      playerDFDict = {}
      playerID = list(team[teamName][i].values())[0]
      playerName = list(team[teamName][i].values())[1]
      career = playercareerstats.PlayerCareerStats(player_id=playerID, timeout=3000)
      careerDF = career.get_data_frames()[0]
      teamID = careerDF[careerDF['SEASON_ID'] == seasonID]['TEAM_ID']
      singleSeasonDetails = shotchartdetail.ShotChartDetail(team_id=teamID, player_id=playerID, season_type_all_star='Regular Season', 
                                                            season_nullable=seasonID, context_measure_simple="FGA").get_data_frames()
      playerDFDict[playerName] = singleSeasonDetails[0]
      statDFList.append(playerDFDict)
      i += 1
    statDFs[teamName] = statDFList

  return statDFs

def drawCourt(ax=None, color="blue", lw=1):
    if ax is None:
      ax = plt.gca()

    # Draw each element
    hoop = Circle((0,0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -12.5), 60, 0, linewidth=lw, color=color)
    outerBox = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    innerBox = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    topFreeThrowArc = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottomFreeThrowArc = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, fill=False)
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Construct three point line
    cornerThreeA = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    cornerThreeB = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    threeArc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Construct half court arc
    centerArcOuter = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    centerArcInner = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    # Add all court elements
    courtElements = [hoop, backboard, outerBox, innerBox, topFreeThrowArc, bottomFreeThrowArc, restricted, cornerThreeA, cornerThreeB, threeArc, centerArcOuter, centerArcInner]

    # Plot each court element to the court
    for element in courtElements:
      ax.add_patch(element)


def shotChart(data, title="", color="b", xDim=(-250, 250), yDim=(422.5, -47.5), lineColor="black", courtColor="white", courtLW=2, 
              flipCourt=False, gridSize=None, ax=None, shotLocation=None):

  if ax == None:
    ax = plt.gca()

  # Court needs to be flipped depending on what side of the court you are looking at
  if not flipCourt:
    ax.set_xlim(xDim)
    ax.set_ylim(yDim)
  else:
    ax.set_xlim(xDim[::-1])
    ax.set_ylim(yDim[::-1])

  # Remove X and Y label and set title
  ax.tick_params(labelbottom="off", labelleft="off")
  ax.set_title(title, fontsize=18)

  # Draw court function: Add all elements to the court
  drawCourt(ax, color=lineColor, lw=courtLW)

  # Seperate missed shots
  xMissed = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_X']
  yMissed = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_Y']

  # Plot missed shots (Color = red, marker=x)
  ax.scatter(xMissed, yMissed, c='r', marker="x", s=150, linewidths=3)

  # Seperate made shots
  xMade = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_X']
  yMade = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_Y']

  # Plot made shots (Color = green, markerr = o)
  ax.scatter(xMade, yMade, facecolors='none', edgecolors='g', marker="o", s=100, linewidths=3)

  if shotLocation != None:
    x = shotLocation[0][0]
    y = shotLocation[0][1]
    ax.scatter(x, y, facecolors='none', edgecolors='b', marker="*", s=250, linewidths=3)

  # Add color to outside of court
  for spine in ax.spines:
    ax.spines[spine].set_lw(courtLW)
    ax.spines[spine].set_color(lineColor)


  return ax

# Get the make and missed with labels for each player 
def getData(statsDF):
  shotsData = {}
  for team in statsDF:
    for player in statsDF[team]:
      playerShots = []
      playerName = list(player.keys())[0]
      # Get only dataFrame for player
      data = list(player.values())[0]
      madeShotsX = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_X']
      madeShotsY = data[data['EVENT_TYPE'] == 'Made Shot']['LOC_Y']
      labelsMade = np.ones(len(madeShotsX))
      madeShots = np.array(list(zip(list(madeShotsX), list(madeShotsY))))

      missedShotsX = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_X']
      missedShotsY = data[data['EVENT_TYPE'] == 'Missed Shot']['LOC_Y']
      labelsMissed = np.zeros(len(missedShotsX))
      missedShots = np.array(list(zip(list(missedShotsX), list(missedShotsY))))
      playerShots = np.concatenate((madeShots, missedShots), axis=0)
      playerLabels = np.concatenate((labelsMade, labelsMissed), axis=0)

      #shotsData[playerName] = np.array(playerShots)
      shotsData[playerName] = [playerShots, playerLabels]

  # Return Format:
  # {PlayerName: [madeShots, labelsMade, missedShots, labelsMissed], PlayerName: [madeShots, labelsMade, missedShots, labelsMissed], ... }
  return shotsData

# Create the KNN models for each of the five player on a team.
# k defaults to 41 but can be passed in as an imput parameter to try different values of k
def KNNModels(shotsDataByPlayer, shotLocation, k=41):
  shotChances = {}
  playerAcc = {}

  # Seperate the dictionary keys to create a list of player names
  playerList = list(shotsDataByPlayer.keys())
  
  # Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[0]][0]
  y = shotsDataByPlayer[playerList[0]][1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

  # Create the KNN  model for the first player 
  playerOneNeigh = KNeighborsClassifier(n_neighbors=k)
  playerOneNeigh.fit(X_train, y_train)

  # Make predictions with X_test
  y_pred = playerOneNeigh.predict(X_test)
  # Check accuarcy of predictions
  playerOneAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[0]] = playerOneAcc

  # Get the shot data for the second plaerys KNN model
  X = shotsDataByPlayer[playerList[1]][0]
  y = shotsDataByPlayer[playerList[1]][1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create the KNN for the second player
  playerTwoNeigh = KNeighborsClassifier(n_neighbors=k)
  playerTwoNeigh.fit(X, y)

  # Make predictions with X_test
  y_pred = playerTwoNeigh.predict(X_test)
  # Check accuarcy of predictions
  playerTwoAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[1]] = playerTwoAcc

  # Get the shot data for the third players KNN model
  X = shotsDataByPlayer[playerList[2]][0]
  y = shotsDataByPlayer[playerList[2]][1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create the KNN for the third player
  playerThreeNeigh = KNeighborsClassifier(n_neighbors=k)
  playerThreeNeigh.fit(X, y)

  # Make predictions with X_test
  y_pred = playerThreeNeigh.predict(X_test)
  # Check accuarcy of predictions
  playerThreeAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[2]] = playerThreeAcc

  # Get the shot data for the fourth players KNN model
  X = shotsDataByPlayer[playerList[3]][0]
  y = shotsDataByPlayer[playerList[3]][1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create the KNN for the fourth player
  playerFourNeigh = KNeighborsClassifier(n_neighbors=k)
  playerFourNeigh.fit(X, y)

  # Make predictions with X_test
  y_pred = playerFourNeigh.predict(X_test)
  # Check accuarcy of predictions
  playerFourAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[3]] = playerFourAcc

  # Get the shot data for the fifth players KNN model
  X = shotsDataByPlayer[playerList[4]][0]
  y = shotsDataByPlayer[playerList[4]][1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Creat the KNN for the fifth player
  playerFiveNeigh = KNeighborsClassifier(n_neighbors=k)
  playerFiveNeigh.fit(X, y)

  # Make predictions with X_test
  y_pred = playerFiveNeigh.predict(X_test)
  # Check accuarcy of predictions
  playerFiveAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[4]] = playerFiveAcc

  # Have each players KNN model predict the % chance they have of making the shot and missing the shot
  playerOneShot = playerOneNeigh.predict_proba(shotLocation)
  playerTwoShot = playerTwoNeigh.predict_proba(shotLocation)
  playerThreeShot = playerThreeNeigh.predict_proba(shotLocation)
  playerFourShot = playerFourNeigh.predict_proba(shotLocation)
  playerFiveShot = playerFiveNeigh.predict_proba(shotLocation)

  # Create a dictionary in the form {"playerName": [miss%, make%], "playerName": [miss%, make%], ...}
  shotChances[playerList[0]] = playerOneShot
  shotChances[playerList[1]] = playerTwoShot
  shotChances[playerList[2]] = playerThreeShot
  shotChances[playerList[3]] = playerFourShot
  shotChances[playerList[4]] = playerFiveShot

  # Return the dictionary containing the player names and their chances of making the shot
  return shotChances, playerAcc

def buildNNModels(shotsDataByPlayer, shotLocation):
  shotChances = {}
  playerAcc = {}

  # Seperate the dictionary keys to create a list of player names
  playerList = list(shotsDataByPlayer.keys())
  
  # Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[0]][0]
  y = shotsDataByPlayer[playerList[0]][1]

  # Split data into 80% training and 20% testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create and train NN model
  playerOneNNModel = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000, learning_rate_init=0.125)
  playerOneNNModel.fit(X_train, y_train)

  # Get predictions based on test locations and check accuracy
  y_pred = playerOneNNModel.predict(X_test)
  playerOneAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[0]] = playerOneAcc

  # Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[1]][0]
  y = shotsDataByPlayer[playerList[1]][1]

  # Split data into 80% training and 20% testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create and train NN model
  playerTwoNNModel = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000, learning_rate_init=0.125)
  playerTwoNNModel.fit(X_train, y_train)

  # Get predictions based on test locations and check accuracy
  y_pred = playerTwoNNModel.predict(X_test)
  playerTwoAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[1]] = playerTwoAcc

  # Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[2]][0]
  y = shotsDataByPlayer[playerList[2]][1]

  # Split data into 80% training and 20% testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create and train NN model
  playerThreeNNModel = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000, learning_rate_init=0.125)
  playerThreeNNModel.fit(X_train, y_train)

  # Get predictions based on test locations and check accuracy
  y_pred = playerThreeNNModel.predict(X_test)
  playerThreeAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[2]] = playerThreeAcc

# Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[3]][0]
  y = shotsDataByPlayer[playerList[3]][1]

  # Split data into 80% training and 20% testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create and train NN model
  playerFourNNModel = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000, learning_rate_init=0.125)
  playerFourNNModel.fit(X_train, y_train)

  # Get predictions based on test locations and check accuracy
  y_pred = playerFourNNModel.predict(X_test)
  playerFourAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[3]] = playerFourAcc

  # Get the shot data for the first players KNN model
  X = shotsDataByPlayer[playerList[4]][0]
  y = shotsDataByPlayer[playerList[4]][1]

  # Split data into 80% training and 20% testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Create and train NN model
  playerFiveNNModel = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000, learning_rate_init=0.125)
  playerFiveNNModel.fit(X_train, y_train)

  # Get predictions based on test locations and check accuracy
  y_pred = playerFiveNNModel.predict(X_test)
  playerFiveAcc = metrics.accuracy_score(y_test, y_pred)

  playerAcc[playerList[4]] = playerFiveAcc

  playerOneShot = playerOneNNModel.predict_proba(shotLocation)
  playerTwoShot = playerTwoNNModel.predict_proba(shotLocation)
  playerThreeShot = playerThreeNNModel.predict_proba(shotLocation)
  playerFourShot = playerFourNNModel.predict_proba(shotLocation)
  playerFiveShot = playerFiveNNModel.predict_proba(shotLocation)

  shotChances[playerList[0]] = playerOneShot
  shotChances[playerList[1]] = playerTwoShot
  shotChances[playerList[2]] = playerThreeShot
  shotChances[playerList[3]] = playerFourShot
  shotChances[playerList[4]] = playerFiveShot

  # Return the dictionary containing the player names and their chances of making the shot
  return shotChances, playerAcc 



# Determine who has the best chane to make the given shot
def whoTakesTheShot(shotChances):
  # Create a list of player names from the keys of the input dictionary
  playerList = list(shotChances.keys())
  # Create a list of player shot % based on the values in the input dictionary 
  shotPercentage = list(shotChances.values())

  maxShot = shotPercentage[0][0][1]
  index = 0
  playerToShoot = {}

  # loop through each player looking for the highest percent chance to make the given shot
  for i in range(len(shotPercentage)):
    if shotPercentage[i][0][1] > maxShot:
      maxShot = shotPercentage[i][0][1]
      index = i
  
  # Return a dictionary in the form {"playerName": [miss%, make%]} of the player who has the highest chance to make the given shot
  playerToShoot[playerList[index]] = shotPercentage[index]

  # Also return the index of where the player with the highest chance to make the shot is in the input dictionary
  return playerToShoot, index

def greatestShot(playerToShootKNN, playerToShootNN, indexKNN, indexNN):
  playerKNN = list(playerToShootKNN.keys())[0]
  playerNN = list(playerToShootNN.keys())[0]

  KNNChane = playerToShootKNN[playerKNN][0][1]
  NNChance = playerToShootNN[playerNN][0][1]

  if KNNChane > NNChance:
    return playerToShootKNN, indexKNN
  else:
    return playerToShootNN, indexNN

if __name__ == "__main__":
  # Get the shotChartDetails of a given team. To change input team, change teamName and playerName to the team name you want to input
  # See top of the page for all supported teams
  teamName = "Los Angeles Lakers"
  statsDF = getPlayerShotChartDetail(teamName=teamName, playerName=LosAngelesLakers)

  # Get the location of all makes and misses by player
  shotsDataByPlayer = getData(statsDF)

  # Get the X and Y location for the shot being predicted
  x = int(input("Input an X value for final shot (-250, 250): "))
  y = int(input("Input a Y value for final shot (-50, 420): "))
  # x = 50
  # y = 125
  shotLocation = [[x, y]]
  #shotLocation = [[5, 65]]
  # Create KNN models for each player and return the % chance they have to amke and miss the shot at the given location
  shotChancesKNN, playerAccKNN = KNNModels(shotsDataByPlayer, shotLocation, k=41)
  shotChancesNN, playerAccNN = buildNNModels(shotsDataByPlayer, shotLocation)
  buildNNModels(shotsDataByPlayer, shotLocation)

  # Determine who has the highest chance to make the shot at the given location
  playerToShootKNN, indexKNN = whoTakesTheShot(shotChancesKNN)

 # Determine who has the highest chance to make the shot at the given location
  playerToShootNN, indexNN = whoTakesTheShot(shotChancesNN)
  playerToShoot, index = greatestShot(playerToShootKNN, playerToShootNN, indexKNN, indexNN)

  # Create a DF for a shot chart
  shotChartDF = statsDF[teamName][index][list(playerToShoot.keys())[0]]
  # Set shot chart title with players name
  shotTitle = list(playerToShoot.keys())[0] + " Shot Chart 2019-20"
  # Create shot chart for player with the highest % chance to make the input shot
  # Players makes are marked with green O's, players misses are marked with red X's, and input shot is marked with a blue star.

  # Prints shot chart of player with the highest percentage to make the shot between KNN and NN
  shotChart(shotChartDF, title=shotTitle, shotLocation=shotLocation)

  playerList = list(shotChancesKNN.keys())
  shotPercentage = list(shotChancesKNN.values())

  print()
  print("Accuracy of KNN:")
  print("Accuracy is tested at 41 neighbors with an 80/20 split on all shots in a season\n")
  for i in range(len(playerAccKNN)):
    string = "Accuracy of KNN model for " + str(playerList[i]) + "\t:  " + str(round(playerAccKNN[playerList[i]], 4))
    print(string)
  print()

  print()
  print("Accuracy of NN:")
  print("Accuracy is tested with an 80/20 split on all shots in a season\n")
  for i in range(len(playerAccNN)):
    string = "Accuracy of KNN model for " + str(playerList[i]) + "\t:  " + str(round(playerAccNN[playerList[i]], 4))
    print(string)
  print()

  # Print results in an easy to read format
  string = "\nPredictions for final shot with KNN at (" + str(x) + ", " + str(y) + "): "
  print(string)
  print()
  
  for i in range(len(playerList)):
    string = str(playerList[i]) + ":\n \t\tProbability of make: " + str(round(shotPercentage[i][0][1], 4)) + " \tProbability of miss: " + str(round(shotPercentage[i][0][0], 4))
    print(string)

  print("\nSuggested player for final shot KNN:")
  string = str(list(playerToShootKNN.keys())[0]) + ": Probability to make the shot: " + str(round(playerToShootKNN[list(playerToShootKNN.keys())[0]][0][1], 4))
  print(string)
  print("\n")

  print("Suggested player for final shot NN:")
  string = str(list(playerToShootNN.keys())[0]) + ": Probability to make the shot: " + str(round(playerToShootNN[list(playerToShootNN.keys())[0]][0][1], 4))
  print(string)
  print("\n")
  
  # Display the shot chart made for the player selected to take the shot
  plt.show()