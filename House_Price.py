import pandas as pd
import warnings
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from helpers.data_prep import *
from helpers.eda import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train=pd.read_csv("dataset/train.csv")
test=pd.read_csv("dataset/test.csv")
df=train.append(test).reset_index(drop=True)
df.head()
"""
* SalePrice - mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalışılan hedef değişkendir.
* MSSubClass: İnşaat sınıfı
* MSZoning: Genel imar sınıflandırması
* LotFrontage: Mülkiyetin sokak ile cephe uzunluğu    ****
* LotArea: Parsel büyüklüğü (fit kare cinsinden)    ****
* Street: Sokak/cadde erişiminin tipi   ****
* Alley: Evin arka cephesindeki bağlantı yolu tipi
* LotShape: Mülkün genel şekli/durumu    *****
* LandContour: Mülkün düzlüğü (tepe olup olmaması)   *****
* Utilities: Mevcut elektrik/su/dogalgaz vb hizmet turleri
* LotConfig: Parsel durumu (ic, dış, iki parsel arası ya da kose gibi)   ****
* LandSlope: Mülkün eğimi   
* Neighborhood: Ames şehir sınırları içindeki fiziksel konumu   ******
* Condition1: Ana yol veya tren yoluna yakınlık   ****
* Condition2: Ana yola veya demiryoluna yakınlık (eğer ikinci bir yol/demiryolu varsa)   ****
* BldgType: Konut tipi   *****
* HouseStyle: Konut sitili    *****
* OverallQual: Genel malzeme ve işçilik kalitesi     
* OverallCond: Konutun genel durum değerlendirmesi
* YearBuilt: Orijinal yapım tarihi
* YearRemodAdd: Yenilenme (elden geçirme, renovasyon) tarihi
* RoofStyle: Çatı tipi
* RoofMatl: Çatı malzemesi
* Exterior1st: Evdeki dış kaplama
* Exterior2nd: Evdeki dış kaplama (birden fazla malzeme varsa)
* MasVnrType: Evin ilave dış duvar kaplama türü (Masonry Veneer : Estetik icin distan örülen ek duvar)
* MasVnrArea: Evin ilave dış duvar kaplama alanı
* ExterQual: Dış malzeme kalitesi
* ExterCond: Dış malzemenin mevcut durumu
* Foundation: Konutun temel tipi
* BsmtQual: Bodrum katin yüksekliği
* BsmtCond: Bodrum katının genel durumu
* BsmtExposure: Bodrumdan bahçenin veya bahçe duvarlarının gorunmesi durumu
* BsmtFinType1: Bodrum katındaki yapılı, badana + zemin olarak tam islem görmüş alanın kalitesi
* BsmtFinSF1: Tam islem görmüş, yapılı alanın metre karesi
* BsmtFinType2: Bodrum katındaki yari yapılı alanın kalitesi (varsa)
* BsmtFinSF2: Bodrumdaki yari yapılı alanın metre karesi
* BsmtUnfSF: Bodrumdaki hiç islem görmemiş alanın metre karesi
* TotalBsmtSF: Bodrum katinin toplam metre karesi
* Heating: Isıtma tipi
* HeatingQC: Isıtma kalitesi ve durumu
* CentralAir: Merkezi klima
* Electrical: Elektrik sistemi
* 1stFlrSF: Birinci Kat metre kare alanı
* 2ndFlrSF: İkinci kat metre kare alanı
* LowQualFinSF: Düşük kaliteli islem/iscilik olan toplam alan (tüm katlar)
* GrLivArea: Zemin katin üstündeki oturma alanı metre karesi
* BsmtFullBath: Bodrum katındaki tam banyolar ( lavabo + klozet + dus + küvet)
* BsmtHalfBath: Bodrum katındaki yarım banyolar ( lavabo + klozet)
* FullBath: Üst katlardaki tam banyolar
* HalfBath: Üst katlardaki yarım banyolar
* BedroomAbvGr: Bodrum seviyesinin üstünde yatak odası sayısı
* KitchenAbvGr: Bodrum seviyesinin üstünde mutfak Sayısı
* KitchenQual: Mutfak kalitesi
* TotRmsAbvGrd: Üst katlardaki toplam oda (banyo içermez)
* Functional: Ev işlevselliği değerlendirmesi
* Fireplaces: Şömine sayısı
* FireplaceQu: Şömine kalitesi
* GarageType: Garajin yeri
* GarageYrBlt: Garajın yapım yılı
* GarageFinish: Garajın iç işçilik/yapim kalitesi
* GarageCars: Garajin araç kapasitesi
* GarageArea: Garajın alanı
* GarageQual: Garaj kalitesi
* GarageCond: Garaj durumu
* PavedDrive: Garajla yol arasındaki bağlantı
* WoodDeckSF: Ustu kapalı ahşap veranda alanı
* OpenPorchSF: Kapı önündeki açık veranda alanı
* EnclosedPorch: Kapı önündeki kapalı veranda alani (muhtemelen ince brandalı)
* 3SsPorch: Üç mevsim kullanılabilen veranda alanı (muhtemelen kis hariç kullanıma uygun, camli kisim)
* ScreenPorch: Sadece sineklik tel ile kapatilmis veranda alanı
* PoolArea: Havuzun metre kare alanı
* PoolQC: Havuz kalitesi
* Fence: Çit kalitesi
* MiscFeature: Diğer kategorilerde bahsedilmeyen çeşitli özellikler
* MiscVal: Çeşitli özelliklerin değeri
* MoSold: Satıldığı ay
* YrSold: Satıldığı yıl
* SaleType: Satış Türü
* SaleCondition: Satış Durumu
"""

#########################################################
# EDA
##########################################################
check_df(df)
test.head()

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
###########################################################
# KATEGORİK DEĞİŞKENLER
##################################
df["Neighborhood"].nunique()
for col in cat_cols:
    cat_summary(df,col)

for col in num_but_cat:
    cat_summary(df,col)

for col in cat_but_car:
    cat_summary(df,col)

###########################################################
# SAYISAL DEĞİŞKENLER
##################################

for col in num_cols:
    num_summary(df,col,plot=True)


######################################
# TARGET ANALIZI
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# target ile bagımsız degiskenlerin korelasyonları
def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

df[["GarageCars","GarageArea"]].corr()

drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature","MSSubClass","MSZoning","LotFrontage","LotArea","Alley","Condition2","RoofMatl","Exterior2nd","MasVnrArea",\
             "BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","Heating","CentralAir","Electrical","LowQualFinSF","BsmtHalfBath","HalfBath","Functional","GarageYrBlt","GarageCars","GarageQual",\
             "GarageCond","PavedDrive","ScreenPorch","Fence","MiscVal","MoSold","3SsnPorch","KitchenAbvGr","PoolArea"]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)

cat_cols = [col for col in cat_cols if col not in drop_list]
num_cols =  [col for col in num_cols if col not in drop_list]
num_but_cat = [col for col in num_but_cat if col not in drop_list]
cat_but_car = [col for col in cat_but_car if col not in drop_list]

df.head(50)
df.shape

######################################
# MISSING_VALUES
######################################

missing_values_table(df)

zero_list=["FireplaceQu","GarageFinish","BsmtExposure","BsmtCond","BsmtQual","BsmtFinType1","MasVnrType"]
df[zero_list]=df[zero_list].apply(lambda x: x.fillna(0), axis=0)

df["GarageType"]=df["GarageType"].fillna("No_garage")
mean_list=["TotalBsmtSF",'BsmtFullBath','GarageArea']
df[mean_list] = df[mean_list].apply(lambda x: x.fillna(x.median()), axis=0)
df["Exterior1st"].fillna("VinylSd",inplace=True)
df['KitchenQual'].fillna("TA",inplace=True)
df['SaleType'].fillna("WD",inplace=True)

######################################
# OUTLIERS
######################################

for col in num_cols:
    replace_with_thresholds(df, col)

df.describe().T

######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################

df.loc[(df["LandContour"] =="Lvl"), "NEW_LAND_CONTOUR"] = 1
df.loc[(df["LandContour"] !="Lvl"), "NEW_LAND_CONTOUR"] = 0

df.loc[(df["YearBuilt"] <= 1914), "NEW_YEAR_BUILT"] = "Hist_B_1"
df.loc[(df["YearBuilt"] > 1914) & (df["YearBuilt"] <= 1960), 'NEW_YEAR_BUILT'] = 'Hist_B_2'
df.loc[(df["YearBuilt"] > 1960) & (df["YearBuilt"] <= 1975), 'NEW_YEAR_BUILT'] = 'Old_B'
df.loc[(df["YearBuilt"] > 1975) & (df["YearBuilt"] <= 1985), 'NEW_YEAR_BUILT'] = '20-30'
df.loc[(df["YearBuilt"] > 1985) & (df["YearBuilt"] <= 1990), 'NEW_YEAR_BUILT'] = '15-20'
df.loc[(df["YearBuilt"] > 1990) & (df["YearBuilt"] <= 1995), 'NEW_YEAR_BUILT'] = '10-15'
df.loc[(df["YearBuilt"] > 1995) & (df["YearBuilt"] <= 2005), 'NEW_YEAR_BUILT'] = '5-10'
df.loc[(df["YearBuilt"] > 2005) & (df["YearBuilt"] <= 2010), 'NEW_YEAR_BUILT'] = 'NEW_B'

df["NEW_DATE_SOLD_RADD"]=df["YrSold"]-df["YearRemodAdd"]

df.loc[df["ExterQual"]=="Ex","ExterQual"]="Gd"
df.loc[df["ExterQual"]=="Fa","ExterQual"]="TA"

df.loc[(df["ExterCond"]=="Fa") | (df["ExterCond"]=="Po"),"ExterCond"]="TA"
df.loc[df["ExterCond"]=="Ex","ExterCond"]="Gd"

df.loc[df["BsmtQual"]=="Ex","BsmtQual"]=5
df.loc[df["BsmtQual"]=="Gd","BsmtQual"]=4
df.loc[df["BsmtQual"]=="TA","BsmtQual"]=3
df.loc[df["BsmtQual"]=="Fa","BsmtQual"]=2
df.loc[df["BsmtQual"]==0,"BsmtQual"]=1

df.loc[df["BsmtCond"]=="Po","BsmtCond"]=1
df.loc[df["BsmtCond"]=="Gd","BsmtCond"]=4
df.loc[df["BsmtCond"]=="TA","BsmtCond"]=3
df.loc[df["BsmtCond"]=="Fa","BsmtCond"]=2
df.loc[df["BsmtCond"]==0,"BsmtCond"]=1

df.loc[df["BsmtExposure"]=="No","BsmtExposure"]=1
df.loc[df["BsmtExposure"]=="Gd","BsmtExposure"]=4
df.loc[df["BsmtExposure"]=="Av","BsmtExposure"]=3
df.loc[df["BsmtExposure"]=="Mn","BsmtExposure"]=2
df.loc[df["BsmtExposure"]==0,"BsmtExposure"]=1

df.loc[df["BsmtFinType1"]=="GLQ","BsmtFinType1"]=5
df.loc[df["BsmtFinType1"]=="ALQ","BsmtFinType1"]=4
df.loc[(df["BsmtFinType1"]=="BLQ")|(df["BsmtFinType1"]=="Rec"),"BsmtFinType1"]=3
df.loc[df["BsmtFinType1"]=="LwQ","BsmtFinType1"]=2
df.loc[df["BsmtFinType1"]=="Unf","BsmtFinType1"]=1

df.loc[df["TotalBsmtSF"]==0,"Total_BSMT"]=1
df.loc[(df["TotalBsmtSF"]>0) & (df["TotalBsmtSF"]<700),"Total_BSMT"]=2
df.loc[(df["TotalBsmtSF"]>=700) & (df["TotalBsmtSF"]<950),"Total_BSMT"]=3
df.loc[(df["TotalBsmtSF"]>=950) & (df["TotalBsmtSF"]<1300),"Total_BSMT"]=4
df.loc[(df["TotalBsmtSF"]>=1300),"Total_BSMT"]=5

df["NEW_BSMT_VALUE"]=df["BsmtFinType1"]+df["BsmtExposure"]+df["BsmtCond"]+df["BsmtQual"]+df["Total_BSMT"]
df["NEW_BSMT_VALUE"]=df["NEW_BSMT_VALUE"].astype(int)
df.loc[(df["BsmtFullBath"]==2) | (df["BsmtFullBath"]==3),"BsmtFullBath"]=1

df.loc[(df["FullBath"]==3)|(df["FullBath"]==4),"FullBath"]=2
df.loc[df["FullBath"]==0,"FullBath"]=1

df.loc[df["FireplaceQu"]==0,"FireplaceQu"]=0
df.loc[df["FireplaceQu"]=="Ex","FireplaceQu"]=5
df.loc[df["FireplaceQu"]=="Gd","FireplaceQu"]=4
df.loc[df["FireplaceQu"]=="TA","FireplaceQu"]=3
df.loc[df["FireplaceQu"]=="Fa","FireplaceQu"]=2
df.loc[df["FireplaceQu"]=="Po","FireplaceQu"]=1

df.loc[df["Fireplaces"]==0,"Fireplaces"]=0
df.loc[df["Fireplaces"]==1,"Fireplaces"]=3
df.loc[df["Fireplaces"]==2,"Fireplaces"]=4
df.loc[df["Fireplaces"]==3,"Fireplaces"]=5
df.loc[df["Fireplaces"]==4,"Fireplaces"]=5

df["NEW_FIREPLACES"]=df["FireplaceQu"]+df["Fireplaces"]
df["NEW_FIREPLACES"]=df["NEW_FIREPLACES"].astype(int)
df["NEW_PORCH"]=df["WoodDeckSF"]+df["OpenPorchSF"]+df["EnclosedPorch"]

df.loc[df["SaleType"]!="WD","SaleType"]="OTHERS"
df.loc[df["SaleCondition"]!="Normal","SaleCondition"]="OTHERS"

drop_list2=["LandContour","YearBuilt","YrSold","YearRemodAdd","Total_BSMT","BsmtFinType1",\
            "BsmtExposure","BsmtCond","BsmtQual","FireplaceQu","Fireplaces","WoodDeckSF","OpenPorchSF","EnclosedPorch"]

for col in drop_list2:
    df.drop(col, axis=1, inplace=True)

######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.04)
df = rare_encoder(df, 0.039)


######################################
# LABEL ENCODING & ONE-HOT ENCODING
######################################

cat_cols = [col for col in df.columns if df[col].dtypes=="O"]
# label encoder ile one hot encoderi birlikte yapıyoruz
df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

######################################
# TRAIN TEST'IN AYRILMASI
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

train_df.to_pickle("DataSet/house_pri/train_df.pkl")
test_df.to_pickle("DataSet/house_pri/test_df.pkl")


#######################################
# MODEL: Random Forests
#######################################

X = train_df.drop(['SalePrice', "Id"], axis=1)
y = train_df["SalePrice"]
#y = np.log1p(train_df['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=46)

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y.mean()

y_pred = rf_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


#######################################
# Model Tuning
#######################################

rf_params = {"max_depth": [10,20,25,30],
             "max_features": [10,25,40],
             "n_estimators": [100,200,500],
             "min_samples_split": [ 3,5,8]}

rf_model = RandomForestRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_
"""{'max_depth': 20,
 'max_features': 50,
 'min_samples_split': 2,
 'n_estimators': 200}"""

"""{'max_depth': 30,
 'max_features': 25,
 'min_samples_split': 3,
 'n_estimators': 200}"""
#######################################
# Final Model
#######################################

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)

y_pred = rf_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = rf_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train, 20)

#######################################
# LightGBM: Model & Tahmin
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred)) # İlkel test hatamız -->27714

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.02,0.01,0.1],
               "n_estimators": [100000,15000],
               "max_depth": [16,20,30],
               "colsample_bytree": [0.2,0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_ # {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 9500}
"""{'colsample_bytree': 0.5,
 'learning_rate': 0.01,
 'max_depth': 5,
 'n_estimators': 10000}"""
#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))  # Model hatamız --> 24539


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_tuned, X_train,30)

