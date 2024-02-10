Instructions for experimentation:
    Set PreyNetwork.PreyNetwork parent class to the appropriate Network.
    Set PredatorNetwork.PredatorNetwork parent class to the appropriate Network.
    Set Globals.PREY_NETWORK_HYPERMARATERS["dimensions"] = <appropriate dimensions list>
    Set Globals.PREDATOR_NETWORK_HYPERMARATERS["dimensions"] = <appropriate dimensions list>
    Comment/uncomment the two flags appropriately at the top of Main.main().
    Run in the terminal:
        python3 main.py --name "<a descriptive name for the serialization file>"
    Cook, cool, and serve