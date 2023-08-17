from pydantic import BaseModel


class Titanic(BaseModel):
    """
    Represents a passenger on the Titanic with various attributes.
    
    Attributes:
        pclass_nan (float): Placeholder for missing values in 'pclass' attribute.
        age_nan (float): Placeholder for missing values in 'age' attribute.
        sibsp_nan (float): Placeholder for missing values in 'sibsp' attribute.
        parch_nan (float): Placeholder for missing values in 'parch' attribute.
        fare_nan (float): Placeholder for missing values in 'fare' attribute.
        sex_male (float): Placeholder for 'sex' attribute being male.
        cabin_Missing (float): Placeholder for 'cabin' attribute being missing.
        cabin_rare (float): Placeholder for 'cabin' attribute being rare.
        embarked_Q (float): Placeholder for 'embarked' attribute being Queenstown.
        embarked_S (float): Placeholder for 'embarked' attribute being Southampton.
        title_Mr (float): Placeholder for 'title' attribute being Mr.
        title_Mrs (float): Placeholder for 'title' attribute being Mrs.
        title_rar (float): Placeholder for 'title' attribute being rare.
    """

    pclass_nan: float
    age_nan: float
    sibsp_nan: float
    parch_nan: float
    fare_nan: float
    sex_male: float
    cabin_Missing: float
    cabin_rare: float
    embarked_Q: float
    embarked_S: float
    title_Mr: float
    title_Mrs: float
    title_rar: float
    
    
