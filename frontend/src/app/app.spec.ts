import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  afterEach(() => {
    TestBed.inject(HttpTestingController).verify();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    TestBed.inject(HttpTestingController)
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the dashboard and online API state', async () => {
    const fixture = TestBed.createComponent(App);
    const request = TestBed.inject(HttpTestingController).expectOne('/api/health');
    request.flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Overview');
    expect(compiled.querySelector('.api-pill')?.textContent).toContain('API connected');
    expect(compiled.textContent).toContain('Assessment workflow');
  });

  it('should show the practice setup and report an unauthenticated start', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();

    const buttons = Array.from(
      fixture.nativeElement.querySelectorAll('button'),
    ) as HTMLButtonElement[];
    const practiceLink = buttons.find((button) =>
      button.textContent?.includes('Practice'),
    ) as HTMLButtonElement;
    practiceLink.click();
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Practice with purpose.');
    (fixture.nativeElement.querySelector('.start-button') as HTMLButtonElement).click();
    const startRequest = http.expectOne('/api/practice/start');
    expect(startRequest.request.body).toEqual({
      subject_id: 'computer-science',
      question_count: 10,
      difficulty: null,
    });
    startRequest.flush(
      { detail: 'Authentication required' },
      { status: 401, statusText: 'Unauthorized' },
    );
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.practice-message')?.textContent).toContain(
      'Sign in as a student',
    );
  });

  it('should bind the selected material subject and submit it with the file', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const materialsLink = (Array.from(
      fixture.nativeElement.querySelectorAll('button'),
    ) as HTMLButtonElement[]).find((button) => button.textContent?.includes('Materials')) as HTMLButtonElement;
    materialsLink.click();
    const subjectsRequest = http.expectOne('/api/subjects');
    subjectsRequest.flush([{ id: 'gis2026', code: 'GIS2026', name: 'GIS AND ITS APPLICATIONS' }]);
    await fixture.whenStable();
    fixture.detectChanges();

    const select = fixture.nativeElement.querySelector('select') as HTMLSelectElement;
    select.value = 'gis2026';
    select.dispatchEvent(new Event('change'));
    const fileInput = fixture.nativeElement.querySelector('.file-field input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [new File(['study material'], 'Module 2.pdf', { type: 'application/pdf' })],
    });
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((resolve) => setTimeout(resolve, 50));
    fixture.detectChanges();

    const uploadButton = fixture.nativeElement.querySelector('.start-button') as HTMLButtonElement;
    expect(uploadButton.disabled).toBe(false);
    uploadButton.click();
    const uploadRequest = http.expectOne('/api/materials/upload');
    expect(uploadRequest.request.body.get('subject_id')).toBe('gis2026');
    uploadRequest.flush({ id: 'material-1', filename: 'Module 2.pdf', status: 'uploaded', size_bytes: 13 });
    await new Promise((resolve) => setTimeout(resolve, 300));
    const processRequest = http.expectOne('/api/materials/material-1/process');
    processRequest.flush({ chunk_count: 1 });
  });

  it('should show the sign-in dialog and report invalid credentials', async () => {
    const fixture = TestBed.createComponent(App);
    const http = TestBed.inject(HttpTestingController);
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();

    (fixture.nativeElement.querySelector('.profile') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('#login-title')?.textContent).toContain(
      'Welcome back.',
    );

    (fixture.nativeElement.querySelector('.login-dialog form') as HTMLFormElement).requestSubmit();
    const loginRequest = http.expectOne('/api/auth/login');
    expect(loginRequest.request.body).toEqual({ email: '', password: '' });
    loginRequest.flush(
      { detail: 'Invalid email or password.' },
      { status: 401, statusText: 'Unauthorized' },
    );
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.login-dialog')?.textContent).toContain(
      'Unable to sign in',
    );
  });
});
