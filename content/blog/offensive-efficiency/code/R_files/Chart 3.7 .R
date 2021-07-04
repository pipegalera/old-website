library(tidyverse)
library(ggtext) 
library(ggrepel)
library(cowplot)
library(extrafont)
#Load only one time
#font_import()
loadfonts(device = "win")
windowsFonts(Times = windowsFont("TT Times New Roman"))

data <- read.csv(file.path("ggplot_data", "chart3.7_lebron_carmelo.csv"),
                     stringsAsFactors = TRUE)

data[data == "TOT"] <- "NYK"


# Teams color palette
teams_palette <- c(
  # Cleveland
  "#860038",
  # Denver
  "#0E2240",
  # Houston
  "#CE1141",
  # Los Angeles Lakers
  "#552583",
  # Miami
  "#98002E",
  # New York Knicks
  "#F58426",
  # Oklahoma City Thunder
  "#007AC1",
  # Portland
  "#E03A3E")



ggplot(data = data, aes(x =Season , y =  OE, color = Player)) +
  geom_line(linetype = 2) +
  geom_point(alpha = 0.6) +
  geom_label_repel(aes(Season ,OE, fill = factor(Tm), label= Tm), 
                  size = 1.5,
                  color = 'white',
                  box.padding = 0.35,
                  segment.color = 'transparent') +
  
  # Colors of the Players and Teams
  scale_fill_manual( values = teams_palette) +
  scale_color_manual(values=c("#56B4E9", "#E69F00")) +
  
  # Title and ticks
  labs(title = "<span style = 'color:#E69F00;'>LeBron James</span> and <span style = 'color:#56B4E9;'>Carmelo Anthony</span> Offensive Efficiency",
       subtitle = "Evolution from the 2003-04 to the 2021 NBA season",
       x = "",
       y = "",
       caption = "Data: Bastekball-reference.com | Plot: @pipegalera") +
  
  # Fix scales
  scale_x_continuous(breaks = seq(2003, 2022, 1)) +
  
  # Theme
  theme_bw() +
  theme(
    # Font
    text = element_text(size = 12, family = "TT Times New Roman"),
    plot.title = element_markdown(size = 16),
    plot.subtitle = element_markdown(size = 12),
    plot.caption = element_markdown(),
    # Grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    panel.border = element_blank(),
    # Remove legend
    legend.position = "none",
    # Years with a 60 degree angle
    axis.text.x = element_text(angle = 60, hjust=1)
    ) 




ggsave(filename = file.path("figures","lebron_carmelo.png"),
       dpi=300, units="in", width = 6.5)





