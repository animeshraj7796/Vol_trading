# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:08:31 2023

@author: animesh.raj
"""


#%%
import pandas as pd
#import straddles_dat as s_dat
import delta_hedge_strats_latest_risk_vega_hedge as d_h_s   #auxilliary functions
import pickle
import numpy as np
callendar={'JAN':'01','FEB':'02','MAR':'03','JUN':'06','SEP':'09','OCT':'10','NOV':'11','DEC':'12','O':'10','N':'11','D':'12'}
import time as time_
import datetime

import numpy as np
# format='%Y-%m-%d%H:%M:%S'
import matplotlib.pyplot as plt
import math
import pdb
#%%
#%%


def current_strat(path,instrument,delta_hedge_strategy_style,tranches,theo,ss,ls,weight,Stoploss_threshhold):


    #%%
    time1=time_.time()


    with open(path, 'rb') as c_file:
            obj1 = pickle.load(c_file)

    obj=obj1.copy()
    #%%
    trad_fil1=[]
    trad_fil1_columns=["Time","Identifier","BuySell","price","ATM_IV","quantity","amountAbsolute"]
    date_tod=list(obj.keys())[0][:10]
    date_tod=datetime.datetime.strptime(date_tod,"%Y-%m-%d")
    date_tod.isoweekday()
    portfolio1={}
    theoretical = round(theo,3)
    tranches=tranches
    short_spread = round(ss,3)
    long_spread= round(ls,3)
    total_size =  100
    trade_size = total_size/tranches
    sell_alpha = False
    totalAmountDeployed = total_size * 180000
    current_portfolio = 0
    trade = 0
    delta_hedge_strategy = {"TIME":,"DISTANCE":, "PORTFOLIOGAMMA":} #time in minutes, custom values
    #if the day is thursday the nhedging style is "TIME"

    #delta_hedge_strategy_style = "DISTANCE"#TIME","DISTANCE","PORTFOLIOGAMMA"]
    instrument = instrument

    lotSizeQuantity={"nifty":50,"banknifty":25}
    loss_threshold=Stoploss_threshhold/weight
    #instrument_exp_name='BANKNIFTY23202'
    wrt=obj[list(obj1.keys())[0]]
    wrt["Time 2 exp(yrs)"].unique()[0]
    wrt=wrt.loc[wrt["Time 2 exp(yrs)"]==min(wrt["Time 2 exp(yrs)"].unique())]
    ert=wrt.loc[wrt['name'].str.startswith(instrument[0].capitalize())]
    prt=ert.loc[ert['Time 2 exp(yrs)']<7/365]
    if instrument== 'banknifty':
        instrument_exp_name=prt.name.iloc[0][:14]
    else:
        instrument_exp_name=prt.name.iloc[0][:10]

    time2=time_.time()
    # print("First",time2-time1)

    time3=time_.time()

    #spread={-1:short_spread,1:long_spread}
    #spreadBuySell = 0.6/100
    counter = 0
    date=list(obj)[0][:10]
    hedge_dist_ratio= #custom values, default 1

    # theoretical = 18
    if date_tod.isoweekday()==4:
        day_counter=352

        #
        #delta_hedge_strategy_style='TIME'



    print("FUNCTION TEST",theoretical,short_spread,long_spread,instrument,instrument_exp_name)
    #%%
    print(theoretical,long_spread,short_spread,loss_threshold)
    #%%
    a = np.linspace(theoretical, theoretical + (tranches + 1) * short_spread, (tranches + 1), endpoint=False)
    b = np.linspace(theoretical - tranches * long_spread, theoretical, tranches, endpoint=False)
    IV_range=np.append(b,a)

    totalQuantityDict = {}
    for j in IV_range:
        if j < theoretical:
           totalQuantityDict[round(j,3)] = trade_size * round(( theoretical -j)/long_spread)

        if j > theoretical :
            totalQuantityDict[round(j,3)] = -1*trade_size * round((j- theoretical )/short_spread)
        if j==theoretical:
            totalQuantityDict[round(j,3)]=0

    print(totalQuantityDict)

    print("FLOATING INTEGER TEST",len(IV_range))
    #%%
    #IV_range=np.arange(theoretical-.04,theoretical+0.04,0.01)
    print(IV_range)
    #%%
    mtmDf = pd.DataFrame(columns=["Time","ATM_strike","ATM_IV","portfolio_delta","portfolio_gamma","portfolio_vega","mtm","risk_5_u_c","risk_10_u_c","risk_20_u_c","risk_5_l_p","risk_10_l_p","risk_20_l_p"])
    #print(mtmDf)
    stopLoss = False

    time4 = time_.time()
    # print("Second", time4 - time3)

    time5 = time_.time()


    for i in sorted(obj.keys()):
        counter = counter +1
        trad_term_in=obj[i]
        #print(trad_term_in)
        tradeTermDf = pd.DataFrame.from_dict(trad_term_in)
        tradeTermDf = tradeTermDf.dropna(subset=['ATM_strike'])
        tradeTermDf = tradeTermDf[tradeTermDf["name"].str.startswith(instrument_exp_name)]
        current_value = tradeTermDf.ATM_IV.iloc[0]
        if current_value>theoretical:
            spreadBuySell=short_spread
        elif current_value<theoretical:
            spreadBuySell=long_spread

        atm_strike = tradeTermDf.ATM_strike.iloc[0]
        fullIdentifier = instrument_exp_name + str(int(atm_strike))
        currentSyntheticFutureATM= trad_term_in.loc[trad_term_in["name"]==fullIdentifier].Syn_Fut.iloc[0]

        #print(i,current_value,theoretical,atm_strike,currentSyntheticFutureATM)
        sigma = abs(current_value -theoretical)//spreadBuySell
        if current_value - theoretical > 0 :
            signSigma = 1
        else:
            signSigma = -1

        #trade opened
        if counter ==1:

            if(len(portfolio1)==0):
                lot=0
                if (current_value -theoretical)> spreadBuySell  :
                    lot= max(-1*trade_size*((current_value -theoretical)//spreadBuySell),-1*total_size)
                elif (theoretical-current_value)> spreadBuySell :
                    lot=min(1*trade_size*((theoretical-current_value)//spreadBuySell),1*total_size)
            #print(lot)
            current_portfolio=current_portfolio+lot
            d_h_s.open_trade_straddle(lot,i,instrument_exp_name,obj,trad_fil1,portfolio1)

            centre_value = theoretical + signSigma * sigma *spreadBuySell


            buy_value = centre_value - 1 *spreadBuySell
            sell_value = centre_value + 1*spreadBuySell

            if current_value > max(IV_range):
                buy_value = max(IV_range) - 1*spreadBuySell
            if current_value < min(IV_range):
                sell_value  = min(IV_range) + 1*spreadBuySell

            if centre_value==theoretical:
                buy_value=theoretical- long_spread
                sell_value= theoretical +short_spread


            LastHedgeSyntheticFutureATM=currentSyntheticFutureATM
            total_gamma_risk=(trad_term_in.loc[trad_term_in["name"]==fullIdentifier].gam_c.iloc[0])*4*total_size*lotSizeQuantity[instrument]
            # print("TRUE GAMMA",trad_term_in.loc[trad_term_in["name"]==fullIdentifier].gam_c.iloc[0],total_size,lotSizeQuantity[instrument])
            lastHedgeTime= tradeTermDf.Time.iloc[0]

            #lastHedgeTime=datetime.datetime.strptime(lastHedgeTime, "%Y-%m-%d %H:%M:%S")
            hedge_distance= 100 #default
            d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)
            # print("TRUE_INITIAL2",current_portfolio,current_value,buy_value,sell_value,"gamma",mtmDf.portfolio_gamma.iloc[-1]*lotSizeQuantity[instrument],total_gamma_risk)

            continue


        #pdb.set_trace()
        #print(time_line)

        if counter < day_counter and stopLoss == False:
            if current_value <= buy_value and buy_value >= min(IV_range) and (abs(mtmDf.portfolio_gamma.iloc[-1]*lotSizeQuantity[instrument]) <abs(total_gamma_risk)):
                # print("buy_true1",current_portfolio,current_value,buy_value,sell_value,"gammas",mtmDf.portfolio_gamma.iloc[-1],total_gamma_risk)
                trade = 1

                quantity = trade_size * trade
                sigmaCentre =sigma
                if current_value >theoretical:
                    sigmaCentre =sigma +1

                centre_value = max(theoretical + signSigma * sigmaCentre *spreadBuySell,min(IV_range))
                # print("BUYX", theoretical, signSigma, sigma, spreadBuySell,theoretical + signSigma * sigma * spreadBuySell, max(IV_range), centre_value)
                buy_value = centre_value - 1 *spreadBuySell
                sell_value = centre_value + 1*spreadBuySell

                trade_lot=totalQuantityDict[round(centre_value,3)]
                quantity=(trade_lot - current_portfolio)
                d_h_s.open_trade_straddle(quantity,i,instrument_exp_name,obj,trad_fil1,portfolio1)


                if current_value > max(IV_range):
                    buy_value = max(IV_range) - 1*spreadBuySell
                if current_value < min(IV_range):
                    sell_value  = min(IV_range) + 1*spreadBuySell

                if centre_value==theoretical:
                    buy_value=theoretical- long_spread
                    sell_value= theoretical +short_spread
                current_portfolio = current_portfolio + quantity

                #d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)
                #print("buy_true2",current_portfolio,current_value,buy_value,sell_value)



            if current_value >= sell_value and sell_value <= max(IV_range) and (abs(mtmDf.portfolio_gamma.iloc[-1])*lotSizeQuantity[instrument] <abs(total_gamma_risk)):
                #print("sell_true1",current_portfolio,current_value,buy_value,sell_value,"gammas",mtmDf.portfolio_gamma.iloc[-1],total_gamma_risk)
                trade = -1
                quantity = trade_size * trade
                #d_h_s.open_trade_straddle(quantity,i,instrument_exp_name,obj,trad_fil1,portfolio1)
                sigmaCentre = sigma
                if current_value < theoretical:
                    sigmaCentre = sigma + 1

                centre_value = min(theoretical + signSigma * sigmaCentre *spreadBuySell,max(IV_range))
                #print("SELLX", theoretical, signSigma, sigma, spreadBuySell,theoretical + signSigma * sigma * spreadBuySell, max(IV_range), centre_value)
                buy_value = centre_value - 1 *spreadBuySell
                sell_value = centre_value + 1*spreadBuySell
                trade_lot=totalQuantityDict[round(centre_value,3)]
                quantity=(trade_lot - current_portfolio)
                d_h_s.open_trade_straddle(quantity,i,instrument_exp_name,obj,trad_fil1,portfolio1)

                if current_value > max(IV_range):
                    buy_value = max(IV_range) - 1*spreadBuySell
                if current_value < min(IV_range):
                    sell_value  = min(IV_range) + 1*spreadBuySell
                if centre_value==theoretical:
                    buy_value=theoretical- long_spread
                    sell_value= theoretical +short_spread
                current_portfolio = current_portfolio + quantity
                #d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)
                #print("sell_true2",current_portfolio,current_value,buy_value,sell_value)

        #         tradeInstrument = str(current_value) + "ATM"
        d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)

        currentTime=tradeTermDf.Time.iloc[0]
        #currentTime=datetime.datetime.strptime(currentTime, "%Y-%m-%d %H:%M:%S")
        distance = currentSyntheticFutureATM - LastHedgeSyntheticFutureATM
        timeDelta = (time_.mktime(currentTime.timetuple())-time_.mktime(lastHedgeTime.timetuple()))/60

        
        
        if delta_hedge_strategy_style == "DISTANCE":
            if(abs(distance)> hedge_distance):
                #print(distance,currentSyntheticFutureATM,LastHedgeSyntheticFutureATM,"hedgeNow")
                d_h_s.delta_hedge(i,instrument_exp_name,trad_term_in,trad_fil1,portfolio1)
                LastHedgeSyntheticFutureATM=currentSyntheticFutureATM
                d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)

        if (delta_hedge_strategy_style == "PORTFOLIOGAMMA"):
            if (abs(mtmDf["portfolio_delta"][counter-1]) > abs((mtmDf["ATM_IV"][counter-1]*delta_hedge_strategy["PORTFOLIOGAMMA"])*mtmDf["portfolio_gamma"][counter-1]*100)) :
                #print(distance,currentSyntheticFutureATM,LastHedgeSyntheticFutureATM,"hedgeNow")
                d_h_s.delta_hedge(i,instrument_exp_name,trad_term_in,trad_fil1,portfolio1)
                d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)

        elif (delta_hedge_strategy_style =="TIME"):
            if( timeDelta > delta_hedge_strategy["TIME"]):
                #print(distance,currentSyntheticFutureATM,LastHedgeSyntheticFutureATM,"hedgeNow")
                d_h_s.delta_hedge(i,instrument_exp_name,trad_term_in,trad_fil1,portfolio1)
                d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)
                lastHedgeTime = currentTime





        if mtmDf.mtm.iloc[-1]*lotSizeQuantity[instrument] < loss_threshold or counter>day_counter  :
            #print("STOPLOSSS TRIGGER",mtmDf.mtm.iloc[-1]*lotSizeQuantity[instrument],mtmDf.mtm.iloc[-1],lotSizeQuantity[instrument])
            #this needs to include turnover
            d_h_s.close_trade(trad_term_in, trad_fil1, portfolio1)
            d_h_s.mtm(mtmDf,i,atm_strike,current_value,portfolio1,trad_term_in)

            break

    time6 = time_.time()
    # print("Third", time6 - time5)


    time7=time_.time()


    tradeFileDf = pd.DataFrame(trad_fil1,columns=trad_fil1_columns)

    tradeFileDf["TotalTurnover"] = tradeFileDf["amountAbsolute"]*lotSizeQuantity[instrument]
    tradeFileDf["weighted_TotalTurnover"]=weight*tradeFileDf["TotalTurnover"]
    tradeFileDf["impactCost"] = 0.01 * tradeFileDf["TotalTurnover"]
    tradeFileDf["pnl"] = tradeFileDf["BuySell"]*tradeFileDf["price"]*tradeFileDf["quantity"]*lotSizeQuantity[instrument]
    tradeFileDf["pnlAfterCost"]= tradeFileDf["pnl"] - tradeFileDf["impactCost"]
    tradeFileDf["weighted_pnl_after_cost"]=weight*tradeFileDf["pnlAfterCost"]
    mtmDf["risk_5_u_c"]=mtmDf["risk_5_u_c"]*lotSizeQuantity[instrument]
    mtmDf["risk_10_u_c"]=mtmDf["risk_10_u_c"]*lotSizeQuantity[instrument]
    mtmDf["risk_20_u_c"]=mtmDf["risk_20_u_c"]*lotSizeQuantity[instrument]
    mtmDf["risk_5_l_p"]=mtmDf["risk_5_l_p"]*lotSizeQuantity[instrument]
    mtmDf["risk_10_l_p"]=mtmDf["risk_10_l_p"]*lotSizeQuantity[instrument]
    mtmDf["risk_20_l_p"]=mtmDf["risk_20_l_p"]*lotSizeQuantity[instrument]


    #print("PnlAfterCost:",tradeFileDf["pnlAfterCost"].sum(),"percentageReturn",tradeFileDf["pnlAfterCost"].sum()/totalAmountDeployed)
    #print("MTM match:",tradeFileDf.pnl.sum()/lotSizeQuantity[instrument],mtmDf.mtm.iloc[-1],lotSizeQuantity[instrument],tradeFileDf.pnl.sum())
    #print("PnlCostRatio:",tradeFileDf["pnl"].sum()/tradeFileDf["impactCost"].sum(),tradeFileDf["impactCost"].sum(),tradeFileDf["pnl"].sum())


    abs_path_output="C:/.........."
    tradeFileDf.to_csv(abs_path_output+date + instrument +"tradeFile.csv")
    portfolioDf = pd.DataFrame.from_dict(portfolio1)
    portfolioDf.to_csv(abs_path_output+date + instrument +"portfolioFile.csv")
    mtmDf.to_csv(abs_path_output+date + instrument + "mtmFile.csv")
    print(tradeFileDf["pnlAfterCost"].sum(), tradeFileDf["TotalTurnover"].sum(),tradeFileDf["weighted_pnl_after_cost"].sum(),tradeFileDf["weighted_TotalTurnover"].sum(),mtmDf["risk_5_u_c"].min(),mtmDf["risk_10_u_c"].min(),mtmDf["risk_20_u_c"].min(), mtmDf["risk_5_l_p"].min(), mtmDf["risk_10_l_p"].min(), mtmDf["risk_20_l_p"].min())

    time8=time_.time()
    # print("Fourth",time8-time7)
    return tradeFileDf["pnlAfterCost"].sum(), tradeFileDf["TotalTurnover"].sum(),tradeFileDf["weighted_pnl_after_cost"].sum(),tradeFileDf["weighted_TotalTurnover"].sum(),mtmDf["risk_5_u_c"].min(),mtmDf["risk_10_u_c"].min(),mtmDf["risk_20_u_c"].min(), mtmDf["risk_5_l_p"].min(), mtmDf["risk_10_l_p"].min(), mtmDf["risk_20_l_p"].min()



