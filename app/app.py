from explainerdashboard import ExplainerDashboard

def main():
    db = ExplainerDashboard.from_config("dashboard.yaml")
    db.run(host='0.0.0.0', port=9050)

if __name__ == "__main__":
    main()
