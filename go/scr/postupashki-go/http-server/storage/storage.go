package storage

import (
	"github.com/google/uuid"
)

// @description interface for RamStorage
type Storage interface {
	PostTask(value string) (*uuid.UUID, error)
	GetStatus(key uuid.UUID) (*string, error)
	GetResult(key uuid.UUID) (*string, error)
	Delete(key uuid.UUID) error
}
