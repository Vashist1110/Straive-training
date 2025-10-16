from flask_restful import Resource
from flask import request
from models.bookModels import Book, db

class BooksGETResource(Resource):
    def get(self):
        books = Book.query.all()
        return [book.to_dict() for book in books]

class BookGETResource(Resource):
    def get(self, id):
        book = Book.query.get(id)
        if book:
            return book.to_dict()
        return {"message": "Book not found"}, 404

class BookPOSTResource(Resource):
    def post(self):
        data = request.get_json()
        if not data or 'title' not in data:
            return {"message": "Title is required"}, 400
        new_book = Book(title=data['title'])
        db.session.add(new_book)
        db.session.commit()
        return new_book.to_dict(), 201

class BookPUTResource(Resource):
    def put(self, id):
        data = request.get_json()
        book = Book.query.get(id)
        if not book:
            return {"message": "Book not found"}, 404
        if 'title' in data:
            book.title = data['title']
        db.session.commit()
        return book.to_dict()

class BookDELETEResource(Resource):
    def delete(self, id):
        book = Book.query.get(id)
        if not book:
            return {"message": "Book not found"}, 404
        db.session.delete(book)
        db.session.commit()
        return '', 204
