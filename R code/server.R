

# Define server logic
server <- function(input, output) {
  
  data <- read.csv("Spends Data.csv", check.names = FALSE)
  hyperparams <- read.csv("Hyperparams.csv", check.names = FALSE)
  coefs <- read.csv("Coefs.csv", check.names = FALSE)
  
  ds <- colnames(data)[1]
  data <- subset(data,select = -ds)
  
  paid_media_spends <- colnames(data)
  media_order <- order(paid_media_spends)
  paid_media_spends <- paid_media_spends[media_order]
  data <- data[paid_media_spends]
  
  thetas <- hyperparams %>% select(ends_with("thetas"))
  thetas <- thetas[media_order]
  
  
  alphas <- hyperparams %>% select(ends_with("alphas"))
  alphas <- alphas[media_order]
  
  gammas <- hyperparams %>% select(ends_with("gammas"))
  gammas <- gammas[media_order]
  
  #Geometric Adstock Function
  adstock_geometric <- function(x, theta) {
    stopifnot(length(theta) == 1)
    if (length(x) > 1) {
      x_decayed <- c(x[1], rep(0, length(x) - 1))
      for (xi in 2:length(x_decayed)) {
        x_decayed[xi] <- x[xi] + theta * x_decayed[xi - 1]
      }
      thetaVecCum <- theta
      for (t in 2:length(x)) {
        thetaVecCum[t] <- thetaVecCum[t - 1] * theta
      } # plot(thetaVecCum)
    } else {
      x_decayed <- x
      thetaVecCum <- theta
    }
    inflation_total <- sum(x_decayed) / sum(x)
    return(list(x = x, x_decayed = x_decayed, thetaVecCum = thetaVecCum, inflation_total = inflation_total))
  }
  
  # Computing the inflexions from the gammas and the adstocked series of the media channels
  
  x_trans <- list()
  
  # Define a function that performs geometric adstock transformation
  transform <- function(i) {
    x <- unlist(data[[paid_media_spends[i]]])
    theta <- unlist(thetas[i], use.names = TRUE)
    transform <- adstock_geometric(x, theta = theta)
    transform$x_decayed
  }
  
  # Use lapply to apply the function to each index in the sequence
  x_trans <- lapply(seq_along(paid_media_spends), transform)
  
  names(x_trans) <- paid_media_spends
  x_trans <- data.frame(x_trans)
  gamma_vec <- unlist(gammas, use.names = FALSE)
  
  #Computing the inflexion points for each media
  inflexions <- unlist(lapply(seq(ncol(x_trans)), function(i) {
    c(range(x_trans[, i]) %*% c(1 - gamma_vec[i], gamma_vec[i]))
    
  }))
  
  inflexions <- as.list(inflexions)
  names(inflexions) <- names(gammas)
  
  #Reading the csv which contains the coefficients of the media variables
  coeffs <- coefs$coef
  names(coeffs) <- coefs$rn
  coeffs <- coeffs[media_order]
  
  #Function to compute the response
  fx_objective <- function(x, coeff, alpha, inflexion, x_hist_carryover, get_sum = TRUE) {
    
    # Adstock scales
    xAdstocked <- x + mean(x_hist_carryover)
    # Hill transformation
    if (get_sum == TRUE) {
      xOut <- coeff * sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    } else {
      xOut <- coeff * ((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    }
    return(xOut)
  }
  
  #Function to compute the slope of response
  fx_slope <- function(x, coeff, alpha, inflexion, x_hist_carryover, get_sum = TRUE) {
    
    # Adstock scales
    xAdstocked <- x + mean(x_hist_carryover)
    # Hill transformation
    if (get_sum == TRUE) {
      xOut <- coeff * sum(alpha * inflexion**alpha * xAdstocked**(-alpha-1) * ((1 + inflexion**alpha / xAdstocked**alpha)**-2))
    } else {
      xOut <- coeff * alpha * inflexion**alpha * xAdstocked**(-alpha-1) * ((1 + inflexion**alpha / xAdstocked**alpha)**-2)
    }
    return(xOut)
  } 
  
  plotDF_scurve <- list()
  max_spends_dt <- list()
  avg_spends_dt <- list()
  
  for (i in paid_media_spends) { # i <- channels[i]
    
    #Extracting the maximum spends
    max_spends <- unlist(summarise_all(select(data, any_of(i)), max))
    
    #Extracting the average spends
    avg_spends <- unlist(summarise_all(select(data, any_of(i)), mean))
    
    #Simulating spends based on the maximum 
    get_max_x <- max_spends * 1.2
    simulate_spend <- seq(0, get_max_x, length.out = 100)
    
    #Computing response for simulated spends
    simulate_response <- fx_objective(
      x = simulate_spend,
      coeff = coeffs[i],
      alpha = unlist(alphas[paste0(i,"_alphas")]),
      inflexion = unlist(inflexions[paste0(i,"_gammas")]),
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    #Computing response for maximum spends
    simulate_response_max <- fx_objective(
      x = max_spends,
      coeff = coeffs[i],
      alpha = unlist(alphas[paste0(i,"_alphas")]),
      inflexion = unlist(inflexions[paste0(i,"_gammas")]),
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    #Computing response for average spends
    simulate_response_avg <- fx_objective(
      x = avg_spends,
      coeff = coeffs[i],
      alpha = unlist(alphas[paste0(i,"_alphas")]),
      inflexion = unlist(inflexions[paste0(i,"_gammas")]),
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    
    plotDF_scurve[[i]] <- data.frame(
      paid_media_spends = i, 
      spend = simulate_spend,
      total_response = simulate_response
    )
    
    max_spends_dt[[i]] <- data.frame(
      paid_media_spends = i,
      max_spend = max_spends,
      response = simulate_response_max
    )
    
    avg_spends_dt[[i]] <- data.frame(
      paid_media_spends = i,
      avg_spend = avg_spends,
      response = simulate_response_avg
    )
    
  }
  
  max_spends_dt <- as_tibble(bind_rows(max_spends_dt))
  plotDT_scurve <- as_tibble(bind_rows(plotDF_scurve))
  avg_spends_dt <- as_tibble(bind_rows(avg_spends_dt))
  
  max_spends_dt <- max_spends_dt %>% mutate_if(is.character, str_replace_all, '_', ' ')
  plotDT_scurve <- plotDT_scurve %>% mutate_if(is.character, str_replace_all, '_', ' ')
  avg_spends_dt <- avg_spends_dt %>% mutate_if(is.character, str_replace_all, '_', ' ')
  
  
  output$Saturation_Curve <- renderHighchart({
    
    # Function to generate a unique series ID for each media type
    generate_series_id <- function(media_type) {
      paste0("series_", media_type)
    }
    
    # Add a series ID to the line data
    plotDT_scurve <- plotDT_scurve %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    # Add a series ID to the scatter point data
    max_spends_dt <- max_spends_dt %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    avg_spends_dt <- avg_spends_dt %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    # Unique media types
    media_types <- unique(plotDT_scurve$paid_media_spends)
    
    # Define colors for each media 
    media_colors <- hue_pal()(length(media_types))
    names(media_colors) <- media_types
    
    # The highcharter plot
    
    hc <- highchart() %>%
      hc_chart(type = "line") %>%
      #hc_title(text = "<b> Sales due to Marketing Spends </b>", align = "center") %>%
      hc_xAxis(
        title = list(text = "<b> Marketing Spends </b>", style = list(color = "black")),  # Set x-axis title color 
        labels = list(
          style = list(color = "black", fontWeight = "bold"),  # Set x-axis labels color 
          formatter = JS("function() { return Highcharts.numberFormat(this.value, 0, '.', ','); }")  # Format x-axis labels
        ),  
        lineColor = "black",  # Set x-axis line color 
        gridLineWidth = 0  # Remove y-axis gridlines
      ) %>%
      hc_yAxis(
        title = list(text = "<b> Sales due to Marketing Spends </b>", style = list(color = "black")),
        labels = list(
          style = list(color = "black", fontWeight = "bold"),  # Set y-axis labels color
          formatter = JS("function() { return Highcharts.numberFormat(this.value, 0, '.', ','); }")  # Format y-axis labels
        ),
        lineColor = "black",  # Set y-axis line color
        gridLineWidth = 0  # Remove y-axis gridlines
      ) %>%
      hc_legend(
        itemStyle = list(color = "black")  # Set legend text color to black
      ) %>%
      hc_plotOptions(
        line = list(marker = list(enabled = FALSE))  # Remove dots from the line
      )
    
    # Add line series for each media type
    hc <- hc %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = plotDT_scurve %>% filter(paid_media_spends == media) %>% select(spend, total_response) %>% list_parse2(),
            type = "line",
            name = media,
            id = generate_series_id(media),
            color = media_colors[[media]],
            marker = list(enabled = FALSE)
          )
        })
      )
    
    # Add scatter points for max spends linked to their respective line series
    hc <- hc %>% hc_add_series(
      data = list(),
      type = "scatter",
      name = "Weekly Maximum",
      marker = list(symbol = "circle", radius = 5),
      color = "black"
    ) %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = max_spends_dt %>% filter(paid_media_spends == media) %>% select(max_spend, response) %>% list_parse2(),
            type = "scatter",
            name = paste0(media),
            linkedTo = generate_series_id(media),
            marker = list(symbol = "circle", radius = 4),
            showInLegend = FALSE,
            color = media_colors[[media]]
          )
        })
      )
    
    # Add scatter points for avg spends linked to their respective line series
    hc <- hc %>% hc_add_series(
      data = list(),
      type = "scatter",
      name = "Weekly Average",
      marker = list(symbol = "triangle", radius = 5),
      color = "black"
    ) %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = avg_spends_dt %>% filter(paid_media_spends == media) %>% select(avg_spend, response) %>% list_parse2(),
            type = "scatter",
            name = paste0(media),
            linkedTo = generate_series_id(media),
            marker = list(symbol = "triangle", radius = 4),
            showInLegend = FALSE,
            color = media_colors[[media]]
          )
        })
      )
    
    # tooltip customisation
    hc <- hc %>%
      hc_tooltip(
        formatter = JS(
          "function() {
        var seriesName = this.series.name;
        var xValue = Highcharts.numberFormat(this.x, 0, '.', ',');
        var yValue = Highcharts.numberFormat(this.y, 0, '.', ',');
        return '<b>Media: ' + seriesName + '</b><br/><br/>' +
               '<b>Spend: ' + xValue + '</b><br/><br/>' +
               '<b>Sales: ' + yValue + '</b>';
      }"
        )
      )
    
    
  })
  
  sat_spends_dt <- list()
  
  for (i in paid_media_spends) { # i <- channels[i]
    
    #Extracting the maximum spends
    max_spends <- unlist(summarise_all(select(data, any_of(i)), max))
    
    #Extracting the average spends
    avg_spends <- unlist(summarise_all(select(data, any_of(i)), mean))
    
    max_spend_resp_ratio <- plotDF_scurve[[i]] %>% mutate(ratio = total_response/spend) %>% pull(ratio) %>% max(na.rm = T)
    
    coeff = coeffs[i]
    alpha = unlist(alphas[paste0(i,"_alphas")])
    inflexion = unlist(inflexions[paste0(i,"_gammas")])
    
    slope_inflexion = alpha*coeff/(4*inflexion)
    
    #Simulating spends based on the maximum 
    get_max_x <- max_spends * 10
    simulate_spend <- seq(inflexion, get_max_x, length.out = 1000)
    
    #Computing slope for simulated spends
    simulate_slope <- fx_slope(
      x = simulate_spend,
      coeff = coeff,
      alpha = alpha,
      inflexion = inflexion,
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    simulate_slope_norm = simulate_slope/slope_inflexion
    
    if(max_spend_resp_ratio <= 1){
      index <- which.min(abs(simulate_slope-0.1))
    } else {
      index <- which.min(abs(simulate_slope_norm-0.1))
    }
    
    sat_spends <- simulate_spend[index]
    
    #Computing response for saturated spends
    simulate_resp_sat <- fx_objective(
      x = sat_spends,
      coeff = coeffs[i],
      alpha = unlist(alphas[paste0(i,"_alphas")]),
      inflexion = unlist(inflexions[paste0(i,"_gammas")]),
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    sat_spends_dt[[i]] <- data.frame(
      paid_media_spends = i,
      sat_spend = sat_spends,
      response = simulate_resp_sat
    )
  }
  
  plotDT_scurve_full <- list()
  
  for (i in paid_media_spends) { # i <- channels[i]
    
    #Extracting the maximum spends
    max_spends <- unlist(summarise_all(select(data, any_of(i)), max))
    
    #Extracting the average spends
    avg_spends <- unlist(summarise_all(select(data, any_of(i)), mean))
    
    #Extracting the saturated spends
    sat_spends <- sat_spends_dt[[i]]$sat_spend
    
    #Simulating spends based on the maximum 
    get_max_x <- max(max_spends, sat_spends)*1.2
    simulate_spend <- seq(0, get_max_x, length.out = 100)
    
    #Computing response for simulated spends
    simulate_resp <- fx_objective(
      x = simulate_spend,
      coeff = coeffs[i],
      alpha = unlist(alphas[paste0(i,"_alphas")]),
      inflexion = unlist(inflexions[paste0(i,"_gammas")]),
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    
    plotDT_scurve_full[[i]] <- data.frame(
      paid_media_spends = i, 
      spend = simulate_spend,
      total_response = simulate_resp
    )
    
  }
  
  sat_spends_dt <- as_tibble(bind_rows(sat_spends_dt))
  plotDT_scurve_full <- as_tibble(bind_rows(plotDT_scurve_full))
  
  sat_spends_dt <- sat_spends_dt %>% mutate_if(is.character, str_replace_all, '_', ' ')
  plotDT_scurve_full <- plotDT_scurve_full %>% mutate_if(is.character, str_replace_all, '_', ' ')
  
  print(sat_spends_dt)
  
  output$Saturation_Curve_full <- renderHighchart({
    
    # Function to generate a unique series ID for each media type
    generate_series_id <- function(media_type) {
      paste0("series_", media_type)
    }
    
    # Add a series ID to the line data
    plotDT_scurve_full <- plotDT_scurve_full %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    # Add a series ID to the scatter point data
    max_spends_dt <- max_spends_dt %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    avg_spends_dt <- avg_spends_dt %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    sat_spends_dt <- sat_spends_dt %>%
      mutate(series_id = sapply(paid_media_spends, generate_series_id))
    
    # Unique media types
    media_types <- unique(plotDT_scurve_full$paid_media_spends)
    
    # Define colors for each media 
    media_colors <- hue_pal()(length(media_types))
    names(media_colors) <- media_types
    
    # The highcharter plot
    
    hc <- highchart() %>%
      hc_chart(type = "line") %>%
      #hc_title(text = "<b> Saturation Curve highlighting Saturation Points </b>", align = "center") %>%
      hc_xAxis(
        title = list(text = "<b> Marketing Spends </b>", style = list(color = "black")),  # Set x-axis title color 
        labels = list(
          style = list(color = "black", fontWeight = "bold"),  # Set x-axis labels color 
          formatter = JS("function() { return Highcharts.numberFormat(this.value, 0, '.', ','); }")  # Format x-axis labels
        ),  
        lineColor = "black",  # Set x-axis line color 
        gridLineWidth = 0  # Remove y-axis gridlines
      ) %>%
      hc_yAxis(
        title = list(text = "<b> Sales due to Marketing Spends </b>", style = list(color = "black")),
        labels = list(
          style = list(color = "black", fontWeight = "bold"),  # Set y-axis labels color
          formatter = JS("function() { return Highcharts.numberFormat(this.value, 0, '.', ','); }")  # Format y-axis labels
        ),
        lineColor = "black",  # Set y-axis line color
        gridLineWidth = 0  # Remove y-axis gridlines
      ) %>%
      hc_legend(
        itemStyle = list(color = "black")  # Set legend text color to black
      ) %>%
      hc_plotOptions(
        line = list(marker = list(enabled = FALSE))  # Remove dots from the line
      )
    
    # Add line series for each media type
    hc <- hc %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = plotDT_scurve_full %>% filter(paid_media_spends == media) %>% select(spend, total_response) %>% list_parse2(),
            type = "line",
            name = media,
            id = generate_series_id(media),
            color = media_colors[[media]],
            marker = list(enabled = FALSE)
          )
        })
      )
    
    # Add scatter points for max spends linked to their respective line series
    hc <- hc %>% hc_add_series(
      data = list(),
      type = "scatter",
      name = "Weekly Maximum",
      marker = list(symbol = "circle", radius = 5),
      color = "black"
    ) %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = max_spends_dt %>% filter(paid_media_spends == media) %>% select(max_spend, response) %>% list_parse2(),
            type = "scatter",
            name = paste0(media),
            linkedTo = generate_series_id(media),
            marker = list(symbol = "circle", radius = 4),
            showInLegend = FALSE,
            color = media_colors[[media]]
          )
        })
      )
    
    # Add scatter points for avg spends linked to their respective line series
    hc <- hc %>% hc_add_series(
      data = list(),
      type = "scatter",
      name = "Weekly Average",
      marker = list(symbol = "triangle", radius = 5),
      color = "black"
    ) %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = avg_spends_dt %>% filter(paid_media_spends == media) %>% select(avg_spend, response) %>% list_parse2(),
            type = "scatter",
            name = paste0(media),
            linkedTo = generate_series_id(media),
            marker = list(symbol = "triangle", radius = 4),
            showInLegend = FALSE,
            color = media_colors[[media]]
          )
        })
      )
    
    # Add scatter points for saturated spends linked to their respective line series
    hc <- hc %>% hc_add_series(
      data = list(),
      type = "scatter",
      name = "Saturation Point",
      marker = list(symbol = "square", radius = 5),
      color = "black"
    ) %>%
      hc_add_series_list(
        map(media_types, function(media) {
          list(
            data = sat_spends_dt %>% filter(paid_media_spends == media) %>% select(sat_spend, response) %>% list_parse2(),
            type = "scatter",
            name = paste0(media),
            linkedTo = generate_series_id(media),
            marker = list(symbol = "square", radius = 4),
            showInLegend = FALSE,
            color = media_colors[[media]]
          )
        })
      )
    
    # tooltip customisation
    hc <- hc %>%
      hc_tooltip(
        formatter = JS(
          "function() {
        var seriesName = this.series.name;
        var xValue = Highcharts.numberFormat(this.x, 0, '.', ',');
        var yValue = Highcharts.numberFormat(this.y, 0, '.', ',');
        return '<b>Media: ' + seriesName + '</b><br/><br/>' +
               '<b>Spend: ' + xValue + '</b><br/><br/>' +
               '<b>Sales: ' + yValue + '</b>';
      }"
        )
      )
    
    
  })
  
}