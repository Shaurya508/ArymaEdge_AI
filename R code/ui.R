#setwd("D:/Febin/Kidde Project/ArymaEdge - Kidde files/Spend Turn-off Simulator")

# install.packages("shiny")
# install.packages("shinythemes")
# install.packages("dplyr")
# install.packages("highcharter")
# install.packages("stringr")
# install.packages("purrr")
# install.packages("jsonlite")
# install.packages("readxl")
# install.packages("reshape2")

library(shiny)
library(shinythemes)
library(dplyr)
library(highcharter)
library(stringr)
library(purrr)
library(jsonlite)
library(readxl)
library(reshape2)
library(scales)

# Define UI for application
ui <- fluidPage(
  theme = shinytheme("cosmo"),
  # tags$li(
  #   class = "dropdown",
  #   a(
  #     img(src = "https://www.arymalabs.com/wp-content/uploads/2022/04/logo1-1.jpg", height = "50px", width = "auto")
  #     #href = "https://www.arymalabs.com/wp-content/uploads/2022/04/logo1-1.jpg"  # Replace with your image link
  #     #target = "_blank"
  #   )
  # ),
  tags$head(
    tags$style(HTML('
      .title-center {
        text-align: center;
        margin-bottom: 0px;
      }
      .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        height: calc(65vh); /* Full viewport height minus some space for title and other elements */
      }
    '))
  ),
  titlePanel(" "
             # tags$div(HTML("<b>Saturation Curves</b>"), style = "text-align: center;")
  ),
  mainPanel(
    div(
      highchartOutput("Saturation_Curve", width = "80%"),
      class = "center-content"
    ),
    div(style = "text-align: center; padding-right: 50px;", 
        tags$h4(tags$b("Saturation Curve highlighting Saturation Points"))),
    div(
      highchartOutput("Saturation_Curve_full", width = "80%"),
      class = "center-content"
    ),
    width = 12  # Make the mainPanel full width
  )
)