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
    try {
      TestBed.inject(HttpTestingController).verify();
    } finally {
      TestBed.resetTestingModule();
    }
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
    const component = fixture.componentInstance as unknown as {
      currentUser: {
        set(user: { id: string; email: string; display_name: string; roles: string[] }): void;
      };
    };
    component.currentUser.set({
      id: 'faculty-1',
      email: 'faculty@example.com',
      display_name: 'Faculty',
      roles: ['faculty'],
    });
    http
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();

    const questionPaperLink = (
      Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[]
    ).find((button) => button.textContent?.includes('Question paper')) as HTMLButtonElement;
    questionPaperLink.click();
    const templatesRequest = http.expectOne('/api/templates');
    templatesRequest.flush([
      {
        id: 'pattern-1',
        name: 'GIS paper',
        question_type: 'MCQ',
        pattern: 'Direct Concept',
        supported_difficulties: ['Easy', 'Medium', 'Hard'],
        supported_bloom_levels: ['Understand'],
        marks: 1,
        total_marks: 1,
        sections: [{ question_type: 'MCQ', pattern: 'Direct Concept', count: 1, marks: 1 }],
      },
    ]);
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

    const generateButton = fixture.nativeElement.querySelector(
      '.start-button',
    ) as HTMLButtonElement;
    expect(generateButton.disabled).toBe(false);
    generateButton.click();
    const uploadRequest = http.expectOne('/api/materials/upload');
    expect(uploadRequest.request.body.get('subject_id')).toBe('gis2026');
    uploadRequest.flush({
      id: 'material-1',
      filename: 'Module 2.pdf',
      status: 'uploaded',
      size_bytes: 13,
    });
    await new Promise((resolve) => setTimeout(resolve, 300));
    const processRequest = http.expectOne('/api/materials/material-1/process');
    processRequest.flush({ chunk_count: 1 });
    await fixture.whenStable();
    const indexRequest = http.expectOne('/api/retrieval/materials/material-1/index');
    indexRequest.flush({ indexed_chunks: 1, embedding_model: 'hashing-v1' });
    await fixture.whenStable();
    const generateRequest = http.expectOne('/api/questions/generate/paper');
    expect(generateRequest.request.body).toEqual({
      template_id: 'pattern-1',
      difficulty: 'Easy',
      bloom_level: 'Understand',
      subject_id: 'gis2026',
      top_k: 5,
    });
    generateRequest.flush([
      {
        id: 'question-1',
        question_text: 'What is a geographic information system?',
        question_type: 'MCQ',
        pattern: 'Direct Concept',
        marks: 1,
        difficulty: 'Easy',
        bloom_level: 'Understand',
        options: [
          { key: 'A', text: 'A mapping system' },
          { key: 'B', text: 'A database only' },
        ],
        explanation: 'A GIS captures, stores, analyzes, and presents geographic data.',
        sources: [{ page_number: 1, text: 'A geographic information system...' }],
      },
    ]);
    await fixture.whenStable();
    fixture.detectChanges();

    expect(
      (fixture.componentInstance as unknown as { activeView: () => string }).activeView(),
    ).toBe('Generated questions');
    expect(fixture.nativeElement.textContent).toContain('What is a geographic information system?');
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
