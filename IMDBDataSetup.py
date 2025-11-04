import pandas as pd

class IMDBDataSetup:
    IMDBDataFrame = pd.DataFrame()

    def importData(fileName):
        IMDBDataSetup.IMDBDataFrame = pd.read_csv(fileName, encoding='utf-8')
        print("Data imported successfully.")
    
    def cleanData():
        IMDBDataSetup.IMDBDataFrame.dropna(inplace=True) # drop rows with missing values
        IMDBDataSetup.IMDBDataFrame.drop_duplicates(inplace=True) # drop duplicate rows
        print("Data cleaned successfully.")

    def createMovieDescription(row):
        title = str(row['Title']) if pd.notna(row['Title']) else 'Unknown'
        director = str(row['Director']) if pd.notna(row['Director']) else 'Unknown'
        genre = str(row['Genre']) if pd.notna(row['Genre']) else 'Unknown'
        star_cast = str(row['Star Cast']) if pd.notna(row['Star Cast']) else 'Unknown'
        duration = str(row['Duration (minutes)']) if pd.notna(row['Duration (minutes)']) else 'Unknown'
        rating = str(row['IMDb Rating']) if pd.notna(row['IMDb Rating']) else 'Unknown'
        year = str(row['Year']) if pd.notna(row['Year']) else 'Unknown'
        
        return f"The Movie '{title}', released in {year}, directed by {director}, is a {genre} film, starring {star_cast}, with a runtime of {duration} minutes and an IMDb rating of {rating}."

    def generateMovieDescriptions():
        IMDBDataSetup.IMDBDataFrame['Movie Description'] = IMDBDataSetup.IMDBDataFrame.apply(IMDBDataSetup.createMovieDescription, axis=1)

