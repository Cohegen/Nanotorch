def wrangle(filepath):
    #reading the csv
    df = pd.read_csv(filepath)

    #applying masking
    mask_federal = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_price = df["price_aprox_usd"] < 100_000
    mask_apartment = df["property_type"] == "apartment"
    df = df[mask_federal & mask_price & mask_apartment]
    
    #removing outliers
    low,high = df["surface_covered_in_m2"].quantile([0.1,0.9])
    mask_area = df["surface_covered_in_m2"].between(low,high)
    df= df[mask_area]

    ##creating a seperate lat and lon columns
    df[["lat","lon"]] = df["lat-lon"].str.split(",",expand=True).astype(float)
    df.drop(columns="lat-lon",inplace=True)

    ##creating a borough feature
    df["borough"] = df["place_with_parent_names"].str.split("|",expand=True)[1]
    df.drop(columns="place_with_parent_names",inplace=True)

    ##dropping columns with more 50 % missing values
    cols = [column for column in df.columns if df[column].isnull().sum() / len(df) > 0.5]
    df.drop(columns=cols,inplace=True)

    ##removing high or low cardinality
    df.drop(columns=[
    "operation",
    "property_type",
    "price",
    "currency",
    "price_aprox_local_currency",
    
    "price_per_m2",
    "properati_url",
],inplace=True)
    
    
    return df