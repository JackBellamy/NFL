import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import re



def empty_list_create():
    """A list of non standard empty values commonly found in datasets"""
    created_empty_list = ['na','NA',' ','-','NAN','NaN']
    return created_empty_list


def find_empty(df):
    """Find non standard NaN"""
    empty_list = empty_list_create()
    found_list = []
    column_list = []
    for c in df.columns:
        found = df[c].isin(empty_list).sum()
        found_list.append(found)
        column_list.append(c)
    found_nan = pd.DataFrame(found_list,index=column_list,columns=['empty_found'])
    return found_nan


def len_min_max(data_df):
    """
    A function that accepts a dataframe and return the maximum and minimum length of string columns
    """
    list_for_df = []
    column_list = []
    for c in data_df.columns:
        if data_df[c].dtypes == "object":
            data_df["len"] = data_df[c].str.len()
            max_str_len = max(data_df["len"])
            min_str_len = min(data_df["len"])
            output = [min_str_len,max_str_len]
            list_for_df.append(output)
            column_list.append(c)
    output = pd.DataFrame(list_for_df,index=column_list,columns=['Min Text Length',"Max Text Length"])
    return output


def set_nan(df):
    """Replace non standard NaNs with np.NaN"""
    empty_list = empty_list_create()
    replaced_df = df.replace(empty_list,np.NaN)
    return replaced_df


def total_duplicates(df):
    """Return the number of duplicates in a dataframe"""
    total_dupes = df.duplicated().sum()
    return total_dupes


def convert_to_pattern(text):
    """
    A function that takes a text string and convert it to an Alpha numeric pattern e.g. text123 --> A4N3
    """
    alpha,alpha_n = re.subn("[a-zA-Z]","A",text)
    output, num_n = re.subn("[0-9]","N",alpha)
    text = ""
    for i in range(len(output)):
        if i == 0:
            letter_count = 0
            final_letter_count = ""

        elif output[i] == output[i-1]:
            letter_count = letter_count + 1

        else:
            if output[i-1] == "A" or output[i-1] == "N":
                final_letter_count = letter_count + 1
                letter_count = 0
            else:
                final_letter_count = ""
                letter_count = 0

        if letter_count == 0:
            output_letter = output[i]
            text = text+str(final_letter_count)+output_letter

        if i == len(output)-1:
            text = text + str(letter_count+1)
    return output,text


def pattern_viz(df_column):
    """
    A function that visuaslises the alpha numeric codes for a given variable
    """
    var_df = pd.DataFrame(df_column)
    var_df['pattern'] = var_df.apply(lambda x: convert_to_pattern(str(x[df_column.name])),axis=1)
    var_df[['pattern','pattern_num']] = pd.DataFrame(var_df["pattern"].to_list(), index= df_column.index)
    grouped_pattern = var_df["pattern_num"].value_counts()
    grouped_pattern.plot(kind="bar",title="Patterns - Count of unique values",ylabel='Frequency',xlabel='Pattern',color='purple')
    plt.show()


def var_value_len(df_column):
    """
    A function that visualizes the length of all the values for a given variable
    """
    try:
        #Calculate lengths
        var = df_column.copy()
        var["var_len"] = var.str.len()
        grouped_var = var["var_len"].value_counts()
        grouped_df = pd.DataFrame(grouped_var)

        #Visualize lengths
        ax = grouped_df.plot(kind="bar",title="Length of values in column",ylabel='Frequency',xlabel='Length')
        for i, each in enumerate(grouped_df.index):
            for col in grouped_df.columns:
                y = grouped_df.loc[each][col]
                ax.text(i, y, y)
        plt.show()
    except:
        print("Value length function does not support int/float/date data types.")


def unique_value_dist(df_column):
    """
    A function that visualizes the frequency of unique values for a given variable
    """
    var = df_column.copy()
    grouped_var = var.value_counts()
    grouped_var.plot(kind="bar",title="Count of unique values",ylabel='Frequency',xlabel='Value',color="blue")
    plt.show()

    
# def string_matching(df_column):
#     """
#     A function that matches 2 strings, intended to help find spelling mistakes etc in the values odf a variable
#     """
#     from fuzzywuzzy import process, fuzz
#     #Create empty dataframe
#     output_df = pd.DataFrame(columns = ['original_string','matched_string','score'])
    
#     #Select unique values
#     var = df_column.copy()
#     input_data = var.value_counts()
    
#     #Loop through values to find similiar values
#     for x in input_data.index:
#         data =  process.extract(x,input_data.index,scorer=fuzz.token_set_ratio)
#         data_sdf = pd.DataFrame(data,columns=('matched_string','score'))
#         data_sdf['original_string'] = x
#         data_sdf = data_sdf[['original_string','matched_string','score']]
#         output = data_sdf.loc[(data_sdf.score < 100) & (data_sdf.score > 90)]
#         output_df = pd.concat([output_df,output])
#     output_df['sort'] = np.minimum(output_df.original_string,output_df.matched_string)
#     output_df["similarity_score"] = output_df['sort']
#     output_df = output_df.loc[output_df.original_string == output_df.sort].reset_index()[["original_string","matched_string","similarity_score"]]
#     return output_df

        
def column_quality_check(df_column,value_len=1,pattern=1,unique_dist=1,similiar_val=1,top_n=0):
    """
    Combine 4 existing univariate functions to create a single quality report.
    Set flags to 0 if you do not want to see a specific visualization.
    similiar_val wil take a while to calculate for big values, itt might be gbest to run by itself.
    """
    
    def return_top_n(series,n=20):
        data_temp = pd.DataFrame(series.value_counts())
        data_temp.rename(columns = {series.name:'Total'}, inplace = True)
        data_temp["Desc_Rank"] = data_temp["Total"].rank(ascending = 0)
        data_temp = data_temp.loc[data_temp.Desc_Rank <= n]
        return series[series.isin(data_temp.index)]
    
    print('\033[1m',"Column - Quality Overview",'\033[0m','\n')
    
    if top_n > 0:
        df_column = return_top_n(df_column,top_n)
    
    if df_column.dtypes == 'O':
        #plt.figure(figsize=(15,6))
        option_sum = value_len + unique_dist + pattern
        if option_sum == 1:
            fig, (ax1) = plt.subplots(1,1, figsize=(13,4))
            
        if option_sum == 2:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,4))
            
        if option_sum == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,4))

        if value_len == 1:
            var_value = var_value_len2(df_column)
            vl = sns.barplot(x = var_value.index, 
                        y = var_value, 
                        ax=ax1,
                        order=var_value.index,
                        color='#003879'
                       )
            for i in vl.containers:
                vl.bar_label(i,)
            vl.set(xlabel='Length', ylabel='Frequency',title='Length of values in column')
            vl.tick_params(axis='x', rotation=90)

        if unique_dist == 1:
            unique_value, top_5, bot_5 = unique_value_dist2(df_column,compress=1)
            uv = sns.barplot(x = unique_value.index,
                        y = unique_value,
                        order=unique_value.index,
                        color="#76C3FF",
                        ax=ax2)
            uv.set(xlabel='Value', ylabel='Frequency',title='Count of unique values')
            uv.tick_params(axis='x', rotation=90)
            print("Top 5 values: ",top_5,"\n")
            print("Bottom 5 values: ",bot_5,"\n")
            

        if pattern == 1:
            pattern_data = pattern_viz2(df_column)
            uv = sns.barplot(x = pattern_data.index,
                        y = pattern_data,
                        order= pattern_data.index,
                        color="#F04C2B",
                        ax=ax3)
            uv.set(xlabel='Pattern', ylabel='Frequency',title='Patterns - Counts of unique values')
            uv.tick_params(axis='x', rotation=90)

#         if similiar_val == 1:
#             print("Possible mispelled entries")
#             return string_matching(df_column)

        plt.tight_layout()
        plt.show()
    else:
        print("Data type not supported")
        
        
def stacked_barplot(df,x_axis, y_axis, colour_by, title="Stacked Barplot"):
    """Function to plot the traditional stacked bar chart"""
    #Configure graph
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (12, 6) #Set graph size 2:1 ratio
    plt.rc('font', size=12)  #Set label, axis etc font size
    plt.rc('axes', titlesize=20) #Set title font size
    plt.rcParams["font.family"] = "montserrat" #Set font type
    cmap = ["#002470","#E5005F","#76C3FF","#0055FF","#F04C2B","#E5F1FF","turquoise","blueviolet","hotpink","dimgrey"]
    
    #Calculate and plot colour by chart
    df = df[[x_axis.name,y_axis.name,colour_by.name]]
    df = df.pivot_table(index=x_axis.name,  
                        columns=colour_by.name, 
                        values=y_axis.name, 
                        aggfunc="sum")
    df = df.reindex(df.sum(axis=1).sort_values(ascending=False).index)
    df.plot.bar(stacked=True, 
                title=title,
                ylabel=y_axis.name, 
                color=cmap)
    plt.show()
    
    #Calculate and plot distribution chart
    df = df.fillna(0)
    df = df.apply(lambda x: x*100/sum(x),axis=1)
    df.plot.bar(stacked=True, 
                title=title,
                ylabel="Percent Contribution",
                legend=0, 
                color=cmap)
    plt.show()
    

def iqr_df_outlier(df,threshold=1.5):
    """Uses IQR to identify outliers in int df columns"""
    int_df = df.select_dtypes(include=np.number)
    outlier_list = []
    column_list = []
    for c in int_df.columns:
        data = int_df[c]
        sorted(data)
        Q1,Q3 = np.percentile(data,[25,75])
        IQR = Q3 - Q1
        lower_range = Q1 - (threshold * IQR)
        upper_range = Q3 + (threshold * IQR)
        output = len([i for i in data if i < lower_range or i > upper_range])
        outlier_list.append(output)
        column_list.append(c)
    outlier_list = pd.DataFrame(outlier_list,index=column_list,columns=['Total_Outliers'])
    return outlier_list


def non_int_min_max(df):
    """
    A function that returns the first and last value of a sorted list as min max
    """
    column_list = []
    value_list = []
    for c in df.columns:
        data_type = df[c].dtypes
        if data_type == "object" or data_type == "datetime64[ns]":
            min_value = df[c].sort_values(ascending=True).iloc[0]
            max_value = df[c].sort_values(ascending=False).iloc[0]
            column_list.append(c)
            value_list.append([min_value,max_value])
    min_max_string_df = pd.DataFrame(value_list,columns=["min_string","max_string"],index=column_list)
    return min_max_string_df


def dq_check(dftemp,verbose=1):
    """Run a number of checks on the provided dataframe"""
    pd.options.display.float_format = "{:,.3f}".format
    
    dfDQAnalysis = pd.DataFrame(list(dftemp.columns.values))
    dfDQAnalysis.columns=['Column_Names'] 
    dfDQAnalysis.set_index('Column_Names', inplace=True)
    
    #Create columns
    dfDQAnalysis['Data_Type'] = list(dftemp.dtypes)
    dfDQAnalysis['Populated_Count'] = list(dftemp.count())
    dfDQAnalysis['Null_Count'] = list(dftemp.isnull().sum())
    dfDQAnalysis['%_Populated'] = dfDQAnalysis.Populated_Count/(dfDQAnalysis.Populated_Count + dfDQAnalysis.Null_Count) * 100
    
    #Check blanks
    dq_blank =  pd.DataFrame(dftemp[dftemp== ''].count(),columns=['Blank'])
    dfDQAnalysis = pd.concat((dfDQAnalysis,dq_blank), axis=1, sort=False)
    dfDQAnalysis['%_Populated_Non_Blank'] = (dfDQAnalysis.Populated_Count - dfDQAnalysis.Blank)/(dfDQAnalysis.Populated_Count) * 100
    

    # Check for single spaces in data
    dq_singlespace =  pd.DataFrame(dftemp[dftemp== ' '].count(),columns=['SingleSpace'])
    dfDQAnalysis = pd.concat((dfDQAnalysis,dq_singlespace), axis=1, sort=False)
    dfDQAnalysis['%_Populated_Non_Space'] = (dfDQAnalysis.Populated_Count - dfDQAnalysis.SingleSpace)/(dfDQAnalysis.Populated_Count) * 100
    
    # Get count of unique values, min, max and join 
    dfDQAnalysis = pd.concat((dfDQAnalysis, dftemp.apply(pd.Series.nunique).rename('Unique_Values')), axis=1)
    
    
    #Add mean, mode, median, sd.dev, coefficient of variance
    if verbose == 1:
        dq_described = dftemp.describe().transpose() 
        dfDQAnalysis = pd.concat((dfDQAnalysis,dq_described), axis=1, sort=False)
        
        #calc mode
        mode_list = []
        columnn_list = []
        for c in dftemp.columns:
            mode = dftemp[c].value_counts().index[0]
            columnn_list.append(c)
            mode_list.append(mode)
        dq_mode = pd.DataFrame(mode_list,index=columnn_list,columns=['mode'])
        dfDQAnalysis = pd.concat((dfDQAnalysis,dq_mode), axis=1, sort=False)
        dfDQAnalysis["CoV"] = dfDQAnalysis["std"]  / dfDQAnalysis["mean"] 
        
        #calc outliers
        outliers = iqr_df_outlier(dftemp)
        dfDQAnalysis = dfDQAnalysis.join(outliers)
        
        #Add string and date minmax
        dq_string_min_max = non_int_min_max(dftemp)
        dfDQAnalysis = dfDQAnalysis.join(dq_string_min_max)
        dfDQAnalysis["min"].fillna(dfDQAnalysis["min_string"], inplace=True)
        dfDQAnalysis["max"].fillna(dfDQAnalysis["max_string"], inplace=True)
        
        #Calc min and max object lengths
        object_length = len_min_max(dftemp)
        dfDQAnalysis = dfDQAnalysis.join(object_length)
        
        dfDQAnalysis = dfDQAnalysis.drop(["count","min_string","max_string"],axis=1)
        dfDQAnalysis.rename(columns={'mean':'Mean','std':'Standard_Deviation','min':'Minimum_Value','25%':'Quartile 1','50%':'Median','75%':'Quartile 3','max':'Max_Value', 'CoV':'Coefficeint Of Variance'},inplace=True)
    return dfDQAnalysis


def iqr_outlier_list(x,threshold=1.5):
    """
    Apply simple IQR outlier detection to a feature and return  a list of all outliers values
    
    """
    sorted(x)
    Q1,Q3 = np.percentile(x,[25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (threshold * IQR)
    upper_range = Q3 + (threshold * IQR)
    output = [i for i in x if i < lower_range or i > upper_range]
    return output


def df_prop(df):
    """
    Standard Datacom EDA function
    """
    row, col = df.shape
    five_num = df.describe().apply(lambda s: s.apply('{0:.5f}'.format))
    print(f"Rows: {row}    Columns: {col}    Total Data Points: {df.size}" )
    print()
    print(df.info())
    return five_num

def view_csv_encoding(data):
    """
    View encoding of a csv file
    """
    with open(data) as file:
        encoding = file.encoding
        return encoding
    
    
def format_plot(rectangle=True):
    """
    A function used to standardize the format of a chart
    """
    plt.style.use('ggplot')
    if rectangle == 1:
        plt.rcParams["figure.figsize"] = (10, 5)
    else:    
        plt.rcParams["figure.figsize"] = (8, 8)
        
    plt.rc('font', size=12)  #Set label, axis etc font size
    plt.rc('axes', titlesize=20) #Set title font size
    plt.rcParams["font.family"] = "montserrat" #Set font type
    cmap = ["#002470","#E5005F","#76C3FF","#0055FF","#F04C2B","#E5F1FF","turquoise","blueviolet","hotpink","dimgrey"]
    return cmap

    
def radar_plot(df,idx,title):
    """
    A function that accepts a traditional dataframe and returns a radar plot for each row based on the column values given
    """
    #format plot
    cmap = format_plot(rectangle=0)
    plt.rcParams["figure.figsize"] = (8, 8)
    #Create required variables
    unique_plot = idx.unique()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    col = 0
    
    #Plot figure
    for i in unique_plot:
        #Filter out ID
        plot_df = df.loc[idx == i]
        
        #Transform values to array and calc theta to get the angles on the chart
        plot_sub_df = plot_df.drop([idx.name], axis= 1)
        labels = plot_sub_df.columns.to_list()
        values = plot_sub_df.iloc[0]
        theta = np.arange(len(values) + 1) / float(len(values)) * 2 * np.pi
        values = np.append(values, values[0])
        
        #Plot graph
        l1 = ax.plot(theta, values, color=cmap[col], marker="o", label=i)
        plt.xticks(theta[:-1], labels, color='grey', size=14)
        ax.tick_params(pad=30)
        ax.axes.yaxis.set_ticklabels([]) # Removes values axis
        ax.legend(bbox_to_anchor=(1.2,1),loc="upper right")
        col += 1
    plt.title(title)
    plt.show()


def three_dim_cluster_viz(df,clusters,x_axis,y_axis,z_axis,x_label="X Label",y_label="Y Label",z_label="Z Label",title="Title",y_angle=30,x_angle=300 ):
    """
    Create a 3D scatter plot that colours by a specific category. Mostly used for visualising RFM clusters.
    """
    #Configure graph
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (12, 12) #Set graph size 2:1 ratio
    plt.rc('font', size=12)  #Set label, axis etc font size
    plt.rc('axes', titlesize=20) #Set title font size
    plt.rc('font', size=14)
    plt.rcParams["font.family"] = "montserrat" #Set font type
    cmap = ["#002470","#E5005F","#76C3FF","#0055FF","#F04C2B","#E5F1FF","turquoise","blueviolet","hotpink","dimgrey"]
    fig = plt.figure()
    
    x = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    #fig = plt.figure(figsize=(10, 6))
    #ax = Axes3D(fig, auto_add_to_figure=False)
    #fig.add_axes(ax)

    #labels = np.unique(df[clusters])
    cluster_count = df[clusters].nunique()
    #clus_name = df[clusters].unique()
    clus_num = [0,1,2,3,4,5,6,7,8,9]
    #filtrered_cmap = cmap[:cluster_count]

    #colour_map_filtered = zip(clus_name,filtrered_cmap)

    #for label, color in colour_map_filtered:
    #    plot_df = df[df[clusters] == label]
    #    ax.scatter(plot_df[x_axis],
    #            plot_df[y_axis],
    #            plot_df[z_axis],  
    #            color=color,  
    #            label=label)

    colour_map = dict(zip(clus_num,cmap))
    colour_map_filtered = dict()
    for key, value in colour_map.items():
        if key <= cluster_count:
            colour_map_filtered[key] = value
    label_color = [colour_map_filtered[l] for l in clusters]

    ax.scatter( x_axis,y_axis, z_axis,c=label_color)

    #ax.set_xlabel(x_label)
    #ax.set_ylabel(y_label)
    #ax.set_zlabel(z_label)
    #ax.set_title(title)
    ax.view_init(y_angle,x_angle)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()
    
    #pairplot_df = pd.concat([df[clusters],df[x_axis],df[y_axis] ,df[z_axis]],axis=1 )
    pairplot_df = pd.concat([clusters,x_axis,y_axis ,z_axis],axis=1 )
    label_color_pairplot = cmap[:cluster_count]
    sns.pairplot(pairplot_df, hue=clusters,corner=True, palette=label_color_pairplot)

def four_dim_cluster_viz(clusters, df, cols):
    from mpl_toolkits.mplot3d import Axes3D
    
    #Configure graph

    cmap = ["#002470","#E5005F","#76C3FF","#0055FF","#F04C2B","turquoise","blueviolet","hotpink","dimgrey"]

    cluster_count = clusters.nunique()
    clus_num = [0,1,2,3,4,5,6,7,8,9]
    
    colour_map = dict(zip(clus_num,cmap))
    colour_map_filtered = dict()
    for key, value in colour_map.items():
        if key <= cluster_count:
            colour_map_filtered[key] = value
    label_color = [colour_map_filtered[l] for l in clusters]
    
    label_color_pairplot = cmap[:cluster_count]
    sns.pairplot(df[cols], hue=clusters.name, corner=True, palette=label_color_pairplot)

def table_contribution_split(df, rows="x",columns="y",measure="z",direction="rows"):
    """
    Create a matrix that splits a dataframe between 2 variables and calculate the contribution of one to the other
    """
    df["measure"] = df[measure]
    for_pivot = df.groupby([rows,columns],as_index=0)["measure"].sum()
    if direction == "rows":
        for_pivot["%_total"] = (for_pivot['measure'] / for_pivot.groupby(rows)['measure'].transform('sum'))
    else:
        for_pivot["%_total"] = (for_pivot['measure'] / for_pivot.groupby(columns)['measure'].transform('sum'))

    pivoted = for_pivot.pivot(index=rows ,columns=columns, values="%_total").reset_index()
    if pivoted.isnull().values.any() == 1:
        pivoted = pivoted.fillna(0)
    
    total_count = df.groupby([rows],as_index=0)["measure"].count().rename(columns={"measure":"Number Of Records"})
    output_pivot = pd.merge(total_count,pivoted,on=rows,how="left").sort_values(by="Number Of Records", ascending=False) 
    output_pivot["Number Of Records"] = output_pivot["Number Of Records"].astype("string")

    #Build dictionary to set format
    formatcols = for_pivot[columns].unique().tolist()
    formatdict = {}
    for formatcol in formatcols: formatdict[formatcol] = "{:.2%}"
    
    #Create a heatrmap of the values along a specific axis
    if direction == "rows":
        output_pivot = output_pivot.style.background_gradient(axis=1).format(formatdict)
    else:
        output_pivot = output_pivot.style.background_gradient(axis=0).format(formatdict)

    return output_pivot
