from app import db

class Table(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    manufacturing = db.Column(db.Float)
    technology = db.Column(db.Float)
    real_estate = db.Column(db.Float)
    
    def __init__(self, manufacturing, technology, real_estate):
        self.manufacturing = manufacturing
        self.technology = technology
        self.real_estate = real_estate

    def __repr__(self):
        return '<id {}>'.format(self.id)
